import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf
import pandas as pd
import shapely
import math




def get_solar_panel_power_weight(wing_span, chordlength):
    panels = wing_span/solar_panel_size*chordlength/solar_panel_size
    weight = panels * solar_panel_mass *g
    power = panels * solar_panel_size**2 * sp_power
    return power,weight

def get_airfoil_area(airfoilfile):
    df = pd.read_csv(airfoilfile, sep='\s+', skiprows=1, header=None, names=['x', 'y'])
    df['x'] = df['x']
    df['y'] = df['y'] 
    coordinates = list(df.itertuples(index=False, name=None))
    cross_section_shape = shapely.Polygon(coordinates)
    
    #Find Volume, Weight from Cross Sectional Area, Wingspan, Foam Density
    cross_sectional_area= cross_section_shape.area
    return(cross_sectional_area)

def get_upper_camber_length(airfoil_filepath):
    
    data = pd.read_table(airfoil_filepath, sep='\s+', skiprows=[0], names=['x', 'y'], index_col=False)

    # Convert the dataframe to a list of tuples
    xy_data = list(zip(data['x'], data['y']))

    # Initialize previous_x and filter to get only upper half airfoil data (x values that are decreasing)
    previous_x = 1
    split_index = None
    for i, (x, y) in enumerate(xy_data):
        if x > previous_x:
            split_index = i
            break
        previous_x = x

    # Split the data into upper and bottom halves based on the split index
    upperHalfAirfoilData = xy_data[:split_index]
    dist_array = [
        np.sqrt((upperHalfAirfoilData[i][0] - upperHalfAirfoilData[i + 1][0])**2 +
                (upperHalfAirfoilData[i][1] - upperHalfAirfoilData[i + 1][1])**2)
        for i in range(len(upperHalfAirfoilData) - 1)
    ]

    # Sum all distances to get the upper camber length
    upper_camber_length = np.sum(dist_array)
    return(upper_camber_length)

def calc_CD0_regression(airfoilfile, re_list=np.linspace(1e5,1e6,10)):
    def fit_power_law(x, y):
        #power law was found to fit CD0 vs Re regressions very well r^2 >0.99 for 5 airfoils. Curran Jacobus 12/1/2024
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        
        # Convert to numpy arrays for easier manipulation
        x = np.array(x)
        y = np.array(y)
        
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("All x and y values must be positive for logarithmic fitting.")
        
        # Take the logarithm of x and y
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Perform linear regression in log-log space
        A = np.vstack([log_x, np.ones_like(log_x)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
        
        # Extract the coefficients
        expo_factor = coeffs[0]
        log_prefactor = coeffs[1]
        prefactor = np.exp(log_prefactor)
        
        return float(prefactor), float(expo_factor)

    def lin_interpolate_CD0(df):
        #Linear Interpolates Coefficient of Drag for when Coefficient of Lift is 0 (CD0)

        # Sort by CL in ascending order to ensure proper interpolation
        df = df.sort_values(by='CL').reset_index(drop=True)

        # Find the indices of the CL values just below and above 0
        cl_below = df[df['CL'] < 0].iloc[-1]  # Last value where CL < 0
        cl_above = df[df['CL'] > 0].iloc[0]   # First value where CL > 0

        # Extract the CL and CD values for interpolation
        cl1, cd1 = cl_below['CL'], cl_below['CD']
        cl2, cd2 = cl_above['CL'], cl_above['CD']

        # Perform linear interpolation for CD when CL = 0
        cd_at_cl_zero = cd1 + (0 - cl1) * (cd2 - cd1) / (cl2 - cl1)

        return cd_at_cl_zero

    def get_CD0(reynolds,airfoil_filepath=airfoilfile):
        #Returns three key airfoil parameters in a dictionary

        #Calculate aerodynamic parameters over range of AoA values, saves to dataframe
        aero_data=pd.DataFrame({"Alpha": [], 'CL': [], 'CD': []})
        for aoa in np.arange(-15,15,0.5):
            data_alpha = nf.get_aero_from_dat_file(
                filename=airfoil_filepath,
                alpha=aoa,
                Re = reynolds,
                model_size='xlarge')
            aero_data_row = pd.DataFrame({"Alpha": [aoa], 'CL': [data_alpha["CL"]], 'CD': [data_alpha["CD"]]})
            aero_data=pd.concat([aero_data,aero_data_row])
        #Desired Values Extracted from Data Frame
        CD0 = lin_interpolate_CD0(aero_data)
        return CD0
    
    cd0_list=[]
    for re in re_list:
        cd0_list.append(get_CD0(re,airfoilfile))
    
    prefactor,expofactor = fit_power_law(re_list,cd0_list)
    return prefactor,expofactor

def calc_lift_curve_slope(airfoil_file):
    cl1 = nf.get_aero_from_dat_file(
                filename=airfoil_file,
                alpha=-2,
                Re=3e5,
                model_size='xlarge'
            )["CL"]
    cl2 = nf.get_aero_from_dat_file(
                filename=airfoil_file,
                alpha=5,
                Re=3e5,
                model_size='xlarge'
            )["CL"]
    return (cl2-cl1)/7



## Constants
g= 9.81 #acceleration of gravity
viscosity = 1.78e-5  # viscosity of air [kg/m/s]
density = 1.23  # density of air [kg/m^3]
pi = np.pi
sp_power = 173.2 #W/M2 energy generated per solar panel. 
foam_density = 30.2# Pink Insulation Foam Density kg/m^3
solar_panel_mass = 0.001 #Mass (kg) per solar panel (incl solder)
solar_panel_size = 0.125 #solar panel size (m)
stabilizer_efficiency = 0.8


##Aircraft Parameters
weight_fuselage = 30 #Fuselage weight newtons
cm_cp_distance = 0.05 #Not yet used
#wing_aoa = 3 #Wing aoa (degrees)
drag_area_fuselage = 0.0001 #fuselage fontal area (m^2)
flat_legth_percent = 0.83 #ratio of flat length to chordlength
airfoilfile = "C:\\Users\\curra\\Documents\\Freshman\\Solar_Airplane\\Prototype 3\\NACA23012.dat"
hstab_airfoil_file = "C:\\Users\\curra\\Documents\\Freshman\\Solar_Airplane\\Prototype 3\\AG35.dat"
airfoil_area = get_airfoil_area(airfoilfile)
fuselage_frontal_area = 0.010834279 #fuselage frontal area m^2
fuselage_length = 0.3623 #m 
mean_fuselage_diameter = 0.130 #m
hstab_span = 1 #m
hstab_chordlen = 0.25 #m
hstab_aoa = -3 #hstab aoa in degrees (negative is below horizon)
hstab_weight = 5 #Hstab weight, newtons
cg_hstab_dist = 1.5 # distance between horizontal stabilizer and cg
LE_cg_dist = 0.1 #distance from Leading edge to Cg in m
#target_upper_camber_length = 0.4 #this can be used to fix chordlen, though not reccomended. The loss in efficiency is likely not gained by better SP coverage
cd0_prefactor_hstab, cd0_expofactor_hstab = calc_CD0_regression(hstab_airfoil_file)
cd0_prefactor, cd0_expofactor = calc_CD0_regression(airfoilfile)
stab_lift_curve_slope = calc_lift_curve_slope(hstab_airfoil_file)
wing_lift_curve_slope = calc_lift_curve_slope(airfoilfile)



opti = asb.Opti()  # initialize an optimization environment
##Variables
wingspan = opti.variable(init_guess=2.5)
#hstab_span = opti.variable(init_guess = 1)
#hstab_chordlen = opti.variable(init_guess = 0.25)
chordlen = opti.variable(init_guess=0.4)
airspeed = opti.variable(init_guess=10)  # cruising speed [m/s]
weight = opti.variable(init_guess=50)  # total aircraft weight [N]
wing_aoa = opti.variable(init_guess = 3)

##Derived
aspect_ratio=wingspan/chordlen
wing_area=wingspan*chordlen
hstab_area = hstab_span * hstab_chordlen
Re = (density / viscosity) * airspeed * (wing_area / aspect_ratio) ** 0.5
oswald = 1.78 * (1- 0.045 * aspect_ratio**0.68) -0.64
dynamic_pressure = 0.5 * density * airspeed ** 2 

#Drag Dependencies
    #wing drag
CL = nf.get_aero_from_dat_file(
                filename=airfoilfile,
                alpha=wing_aoa,
                Re=Re,
                model_size='xlarge'
            )["CL"]
CD_wing = CL / nf.get_aero_from_dat_file(
                filename=airfoilfile,
                alpha=wing_aoa,
                Re=Re,
                model_size='xlarge'
            )["CD"]

CD0= cd0_prefactor * (Re ** cd0_expofactor)
drag_parasite_wing =  wing_area * CD0 *dynamic_pressure
drag_induced_wing = wing_area * dynamic_pressure * CL ** 2 / (np.pi * aspect_ratio * oswald)
    #Form Drag (Fuselage)
form_factor = 1+60/((fuselage_length/mean_fuselage_diameter)**3) + 0.0025*fuselage_length/mean_fuselage_diameter
skin_friction_coefficient = 0.455/(np.log(Re)**2)
CD_fuselage = form_factor *skin_friction_coefficient
drag_fuselage = dynamic_pressure * fuselage_frontal_area * CD_fuselage
    #Hstab drag
Re_hstab = (density / viscosity) * airspeed *  hstab_chordlen
CL_hstab = nf.get_aero_from_dat_file(
                filename=hstab_airfoil_file,
                alpha=-hstab_aoa,
                Re=Re_hstab,
                model_size='xlarge'
            )["CL"]
CD_wing_hstab = CL / nf.get_aero_from_dat_file(
                filename=hstab_airfoil_file,
                alpha=-hstab_aoa,
                Re=Re_hstab,
                model_size='xlarge'
            )["CD"]

CD0_hstab= cd0_prefactor_hstab * (Re_hstab ** cd0_expofactor_hstab)
drag_parasite_hstab =  hstab_area * CD0 *dynamic_pressure
drag_induced_hstab = hstab_area * dynamic_pressure * CL_hstab ** 2 / (np.pi * hstab_span * oswald / hstab_chordlen)   
    #totals
drag= drag_induced_wing + drag_fuselage +drag_parasite_wing +drag_induced_hstab +drag_parasite_hstab
lift_cruise = dynamic_pressure * wing_area * CL - (dynamic_pressure * hstab_chordlen * hstab_span * CL_hstab)

##Weight Dependency
_,sp_weight = get_solar_panel_power_weight(wingspan,chordlen)
foam_weight = airfoil_area * chordlen**2 *wingspan * foam_density * g
spar_weight = 2*0.094*wingspan*g
empennage_weight = 0.39*g
weight = sp_weight + foam_weight+weight_fuselage+spar_weight+empennage_weight +hstab_weight
##Power Relationship
power_produced_wing, _ = get_solar_panel_power_weight(wingspan,chordlen)
power_produced_htab, _ = get_solar_panel_power_weight(hstab_span,hstab_chordlen)
power_used = airspeed*drag
power_produced = (power_produced_wing +power_produced_htab)
power_ratio = power_produced / power_used

## Stability
#https://ciurpita.tripod.com/rc/notes/neutralPt.html
aerodynamic_center = 0.25
volume_coefficient = hstab_area * cg_hstab_dist / wing_area / chordlen
neutral_point = aerodynamic_center +stabilizer_efficiency * volume_coefficient * 0.6 * stab_lift_curve_slope/wing_lift_curve_slope
static_margin = neutral_point - LE_cg_dist/chordlen
##Constraints
opti.subject_to(wing_aoa >1)
opti.subject_to(wing_aoa < 5)
#opti.subject_to(wingspan <= 3)
opti.subject_to(airspeed <= 30)
opti.subject_to(weight <= lift_cruise)
opti.subject_to(power_ratio > 1)
#opti.subject_to(hstab_span > 0)
#opti.subject_to(hstab_chordlen > 0)

#Objective
opti.maximize(power_ratio)

sol = opti.solve(max_iter=100)

for value in [
    "power_ratio",
    "power_used",
    "power_produced",
    "airspeed",
    "weight",
    "wingspan",
    "chordlen",
    "wing_aoa",
#    "hstab_span",
#    "hstab_chordlen",
    "static_margin",
    "drag"
]:
    print(f"{value:10} = {sol(eval(value)):.6}")