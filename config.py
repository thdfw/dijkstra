import numpy as np
import configparser
import pendulum

def to_celcius(t):
    return (t-32)*5/9


class DParams():

    def __init__(self) -> None:
        config = configparser.ConfigParser()
        config.read('parameters.conf')
        self.start_time = pendulum.parse(config.get('parameters', 'START_TIME')).set(minute=0, second=0)
        self.horizon = config.getint('parameters', 'HORIZON_HOURS')
        self.num_layers = config.getint('parameters', 'NUM_LAYERS')
        self.storage_volume = config.getfloat('equipment', 'STORAGE_VOLUME_GALLONS')
        self.max_hp_elec_in = config.getfloat('equipment', 'HP_MAX_ELEC_POWER_KW')
        self.min_hp_elec_in = config.getfloat('equipment', 'HP_MIN_ELEC_POWER_KW')
        self.initial_top_temp = config.getfloat('initial', 'INITIAL_TOP_TEMP_F')
        self.initial_thermocline = config.getfloat('initial', 'INITIAL_THERMOCLINE')
        self.storage_losses_percent = config.getfloat('equipment','STORAGE_LOSSES_PERCENT')
        self.dp_csv = config.get('csv_data', 'DP_CSV')
        self.lmp_csv = config.get('csv_data', 'LMP_CSV')
        self.weather_csv = config.get('csv_data', 'WEATHER_CSV')
        self.alpha = config.getfloat('house', 'ALPHA')
        self.beta = config.getfloat('house', 'BETA')
        self.gamma = config.getfloat('house', 'gamma')
        self.no_power_rswt = -self.alpha/self.beta
        self.intermediate_power = config.getfloat('house', 'INTERMEDIATE_POWER')
        self.intermediate_rswt = config.getfloat('house', 'INTERMEDIATE_RSWT')
        self.dd_power = config.getfloat('house', 'DD_POWER')
        self.dd_rswt = config.getfloat('house', 'DD_RSWT')
        self.dd_delta_t = config.getfloat('house', 'DD_DELTA_T') 
        self.cop_intercept = config.getfloat('COP', 'INTERCEPT') 
        self.cop_oat_coeff = config.getfloat('COP', 'OAT_COEFF') 
        self.cop_lwt_coeff = config.getfloat('COP', 'LWT_COEFF') 
        self.now_for_file = round(pendulum.now('UTC').timestamp())
        # Compute quadratic coefficients to estimate heating power from SWT
        x_rswt = np.array([self.no_power_rswt, self.intermediate_rswt, self.dd_rswt])
        y_hpower = np.array([0, self.intermediate_power, self.dd_power])
        A = np.vstack([x_rswt**2, x_rswt, np.ones_like(x_rswt)]).T
        self.quadratic_coefficients = [float(x) for x in np.linalg.solve(A, y_hpower)] 
        self.available_top_temps = self.get_available_top_temps()
        self.min_cop = 1
        self.max_cop = 3

    def COP(self, oat, lwt, fahrenheit=True):
        if fahrenheit: #TODO: will it ever be in celcius?
            oat = to_celcius(oat)
            lwt = to_celcius(lwt)
        return self.cop_intercept + self.cop_oat_coeff*oat + self.cop_lwt_coeff*lwt      

    def required_heating_power(self, oat, ws):
        r = self.alpha + self.beta*oat + self.gamma*ws
        return r if r>0 else 0

    def delivered_heating_power(self, swt):
        a, b, c = self.quadratic_coefficients
        d = a*swt**2 + b*swt + c
        return d if d>0 else 0

    def required_swt(self, rhp):
        a, b, c = self.quadratic_coefficients
        return -b/(2*a) + ((rhp-b**2/(4*a)+b**2/(2*a)-c)/a)**0.5

    def delta_T(self, swt):
        d = self.dd_delta_t/self.dd_power * self.delivered_heating_power(swt)
        d = 0 if swt<self.no_power_rswt else d
        return d if d>0 else 0
    
    def delta_T_inverse(self, rwt):
        a, b, c = self.quadratic_coefficients
        aa = -self.dd_delta_t/self.dd_power * a
        bb = 1-self.dd_delta_t/self.dd_power * b
        cc = -self.dd_delta_t/self.dd_power * c
        return -bb/(2*aa) - ((rwt-bb**2/(4*aa)+bb**2/(2*aa)-cc)/aa)**0.5 - rwt
    
    def get_available_top_temps(self):
        available_temps = [round(self.initial_top_temp)]
        x = round(self.initial_top_temp)
        while round(x + self.delta_T_inverse(x)) <= 175:
            x = round(x + self.delta_T_inverse(x))
            available_temps.append(x)
        while x <= 175:
            x += 10
            available_temps.append(x)
        x = round(self.initial_top_temp)
        while self.delta_T(x) >= 3:
            x = round(x - self.delta_T(x))
            available_temps.append(x)
        while x >= 70:
            x += -10
            available_temps.append(x)
        available_temps = sorted(available_temps)
        return available_temps