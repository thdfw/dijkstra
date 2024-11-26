from config import DParams
import pandas as pd
import csv
import pendulum

def to_kelvin(t):
    return (t-32)*5/9 + 273.15


class DNode():
    def __init__(self, time_slice:int, top_temp:float, thermocline:float, parameters:DParams):
        self.params = parameters
        # Position in graph
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.thermocline = thermocline
        # Dijkstra's algorithm
        self.pathcost = 0 if time_slice==parameters.horizon else 1e9
        self.next_node = None
        # Absolute energy level
        tt_idx = parameters.available_top_temps.index(top_temp)
        tt_idx = tt_idx-1 if tt_idx>0 else tt_idx
        self.bottom_temp = parameters.available_top_temps[tt_idx]
        self.energy = self.get_energy()
        self.index = None #TODO: index

    def __repr__(self):
        return f"Node[time_slice:{self.time_slice}, top_temp:{self.top_temp}, thermocline:{self.thermocline}]"

    def get_energy(self):
        m_layer_kg = self.params.storage_volume*3.785 / self.params.num_layers
        kWh_above_thermocline = (self.thermocline-0.5)*m_layer_kg * 4.187/3600 * to_kelvin(self.top_temp)
        kWh_below_thermocline = (self.params.num_layers-self.thermocline+0.5)*m_layer_kg * 4.187/3600 * to_kelvin(self.bottom_temp)
        return kWh_above_thermocline + kWh_below_thermocline


class DEdge():
    def __init__(self, tail:DNode, head:DNode, cost:float, hp_heat_out:float):
        self.tail = tail
        self.head = head
        self.cost = cost
        self.hp_heat_out = hp_heat_out

    def __repr__(self):
        return f"Edge: {self.tail} --cost:{round(self.cost,3)}--> {self.head}"


class DForecast():
    def __init__(self, params:DParams) -> None:

        start_year = params.start_time.year
        start_date = pendulum.datetime(start_year, 1, 1, 0, 0, tz='America/New_York')
        end_date = pendulum.datetime(start_year, 12, 31, 23, 0, tz='America/New_York')
        time_list = [start_date.add(hours=i) for i in range(int((end_date-start_date).in_hours())+1)]

        with open(params.weather_csv, newline='', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            self.oat, self.ws = zip(*reader)
        self.oat = [float(x) for x in self.oat]        
        self.ws = [float(x) for x in self.ws]        
        with open(params.dp_csv, newline='', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            self.dp = [float(row[0])/10 for row in reader]
        with open(params.lmp_csv, newline='', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            self.lmp = [float(row[0])/10 for row in reader]
        self.elec_price = [dp+lmp for dp,lmp in zip(self.dp,self.lmp)]
        
        if not (len(self.dp)==len(self.lmp)==len(self.oat)==len(time_list)):
            raise ValueError(f"The provided CSV files must contain {len(time_list)} rows.")

        self.load = [params.required_heating_power(oat,ws) for oat,ws in zip(self.oat,self.ws)]
        self.rswt = [params.required_swt(x) for x in self.load]

        max_load_elec = max(self.load) / params.COP(min(self.oat), max(self.rswt))

        if max_load_elec > params.max_hp_elec_in:
            error_text = f"\nOn the coldest hour:"
            error_text += f"\n- The heating requirement is {round(max(self.load),2)} kW"
            error_text += f"\n- The COP is {round(params.COP(min(self.oat), max(self.rswt)),2)}"
            error_text += f"\n=> Need a HP which can reach {round(max_load_elec,2)} kW electrical power"
            error_text += f"\n=> The given HP is undersized ({params.max_hp_elec_in} kW electrical power)"
            raise ValueError(error_text)

        self.time = [x for x in time_list if (x>=params.start_time and x<=params.start_time.add(hours=params.horizon))]
        self.oat = [x for x,y in zip(self.oat, time_list) if y in self.time]        
        self.ws = [x for x,y in zip(self.ws, time_list) if y in self.time]        
        self.dp = [x for x,y in zip(self.dp, time_list) if y in self.time]
        self.lmp = [x for x,y in zip(self.lmp, time_list) if y in self.time]
        self.elec_price = [x for x,y in zip(self.elec_price, time_list) if y in self.time]
        self.load = [x for x,y in zip(self.load, time_list) if y in self.time]
        self.rswt = [x for x,y in zip(self.rswt, time_list) if y in self.time]