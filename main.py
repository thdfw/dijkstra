import time
import pendulum
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from config import DParams
from dtypes import DNode, DEdge, DForecast

class DGraph():

    def __init__(self):
        self.params = DParams()
        self.forecasts = DForecast(self.params)
        print("Creating graph...")
        self.create_nodes()
        self.create_edges()
        print("Solving Dijkstra...")
        self.solve_dijkstra()
        print("Done")
        self.plot()

    def create_nodes(self):
        self.initial_node = DNode(0, self.params.initial_top_temp, self.params.initial_thermocline, self.params)
        self.nodes = {}
        for time_slice in range(self.params.horizon+1):
            self.nodes[time_slice] = [self.initial_node] if time_slice==0 else []
            self.nodes[time_slice].extend(
                DNode(time_slice, top_temp, thermocline, self.params)
                for top_temp in self.params.available_top_temps
                for thermocline in range(1,self.params.num_layers+1)
                if (time_slice, top_temp, thermocline) != (0, self.params.initial_top_temp, self.params.initial_thermocline)
            )
            # TODO: check if you need index
            # self.nodes_by_energy = sorted(self.nodes[time_slice], key=lambda x: (x.energy, x.top_temp), reverse=True)
            # for n in self.nodes[time_slice]:
            #     n.index = self.nodes_by_energy.index(n)+1

    def create_edges(self):
        self.edges = {}
        self.bottom_node = DNode(0,self.params.available_top_temps[0], 0, self.params)
        self.top_node = DNode(0, self.params.available_top_temps[-1], self.params.num_layers, self.params)
        self.energy_between_consecutive_states = (DNode(0,self.params.available_top_temps[0],2,self.params).energy 
                                                  - DNode(0,self.params.available_top_temps[0],1,self.params).energy)

        for h in range(self.params.horizon):
            
            for node_now in self.nodes[h]:
                self.edges[node_now] = []
                
                for node_next in self.nodes[h+1]:

                    # The energy difference between two states might be lower than energy between two nodes
                    losses = self.params.storage_losses_percent/100 * (node_now.energy-self.bottom_node.energy)
                    if self.forecasts.load[h]==0 and losses>0 and losses<self.energy_between_consecutive_states:
                        losses = self.energy_between_consecutive_states + 1/1e9

                    store_heat_in = node_next.energy - node_now.energy
                    hp_heat_out = store_heat_in + self.forecasts.load[h] + losses
                    
                    # This condition reduces the amount of times we need to compute the COP
                    if (hp_heat_out/self.params.min_cop <= self.params.max_hp_elec_in and
                        hp_heat_out/self.params.max_cop >= self.params.min_hp_elec_in):
                    
                        cop = self.params.COP(oat=self.forecasts.oat[h], lwt=node_next.top_temp)

                        if (hp_heat_out/cop <= self.params.max_hp_elec_in and 
                            hp_heat_out/cop >= self.params.min_hp_elec_in):

                            cost = self.forecasts.elec_price[h]/100 * hp_heat_out/cop
                            
                            # Charging the storage
                            if store_heat_in > 0:
                                # Same top temperature, thermocline going down
                                if node_next.top_temp == node_now.top_temp and node_next.thermocline > node_now.thermocline:
                                    self.edges[node_now].append(DEdge(node_now, node_next, cost, hp_heat_out))
                                # Higher top temperature
                                if node_next.top_temp > node_now.top_temp:
                                    self.edges[node_now].append(DEdge(node_now, node_next, cost, hp_heat_out))
                            
                            # Discharging the storage
                            elif store_heat_in < 0:
                                # Check for RSWT
                                if ((hp_heat_out < self.forecasts.load[h] and 
                                     self.forecasts.load[h] > 0)
                                     and
                                    (node_now.top_temp < self.forecasts.rswt[h] or 
                                     node_next.top_temp < self.forecasts.rswt[h])):
                                    # TODO: add a soft constraint (e.g. cost+=1e5)
                                    continue
                                # Same top temperature, thermocline going up
                                if node_next.top_temp == node_now.top_temp and node_next.thermocline < node_now.thermocline:
                                    self.edges[node_now].append(DEdge(node_now, node_next, cost, hp_heat_out))
                                # Top temperature going down
                                if node_next.top_temp < node_now.top_temp:
                                    self.edges[node_now].append(DEdge(node_now, node_next, cost, hp_heat_out))
                            
                            # Keeping the store in the same state
                            else:
                                self.edges[node_now].append(DEdge(node_now, node_next, cost, hp_heat_out))

    def solve_dijkstra(self):
        for time_slice in range(self.params.horizon-1, -1, -1):
            for node in self.nodes[time_slice]:
                best_edge = min(self.edges[node], key=lambda e: e.head.pathcost + e.cost)
                if best_edge.hp_heat_out < 0: 
                    best_edge_neg = max([e for e in self.edges[node] if e.hp_heat_out<0], key=lambda e: e.hp_heat_out)
                    best_edge_pos = min([e for e in self.edges[node] if e.hp_heat_out>=0], key=lambda e: e.hp_heat_out)
                    best_edge = best_edge_pos if (-best_edge_neg.hp_heat_out >= best_edge_pos.hp_heat_out) else best_edge_neg
                node.pathcost = best_edge.head.pathcost + best_edge.cost
                node.next_node = best_edge.head

    def plot(self):
        self.list_toptemps = []
        self.list_thermoclines = []
        self.list_hp_energy = []
        self.list_storage_energy = []
        node_i = self.initial_node
        while node_i.next_node is not None:
            edge_i = [e for e in self.edges[node_i] if e.head==node_i.next_node][0]
            self.list_toptemps.append(node_i.top_temp)
            self.list_thermoclines.append(node_i.thermocline)
            self.list_hp_energy.append(edge_i.hp_heat_out)
            self.list_storage_energy.append(node_i.energy)
            node_i = node_i.next_node
        self.list_toptemps.append(node_i.top_temp)
        self.list_thermoclines.append(node_i.thermocline)
        self.list_hp_energy.append(edge_i.hp_heat_out)
        self.list_storage_energy.append(node_i.energy)
        list_soc = [(x-self.bottom_node.energy)/(self.top_node.energy-self.bottom_node.energy)*100 for x in self.list_storage_energy]
        # Plot the shortest path
        self.params.start_time
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,6))
        begin = self.params.start_time.format('YYYY-MM-DD HH:mm')
        end = self.params.start_time.add(hours=self.params.horizon).format('YYYY-MM-DD HH:mm')
        fig.suptitle(f'From {begin} to {end}\nCost: {round(self.initial_node.pathcost,2)} $', fontsize=10)
        list_time = list(range(len(list_soc)))
        # Top plot
        ax[0].step(list_time, self.list_hp_energy, where='post', color='tab:blue', label='HP', alpha=0.6)
        ax[0].step(list_time, self.forecasts.load, where='post', color='tab:red', label='Load', alpha=0.6)
        ax[0].legend(loc='upper left')
        ax[0].set_ylabel('Heat [kWh]')
        if max(self.list_hp_energy)<20:
            ax[0].set_ylim([-0.5,20]) 
        ax2 = ax[0].twinx()
        ax2.step(list_time, self.forecasts.elec_price, where='post', color='gray', alpha=0.6, label='Elec price')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('Electricity price [cts/kWh]')
        if min(self.forecasts.elec_price)>0: ax2.set_ylim([0,60])
        # Bottom plot
        norm = Normalize(vmin=self.params.available_top_temps[0], vmax=self.params.available_top_temps[-1])
        cmap = matplotlib.colormaps['Reds']
        tank_top_colors = [cmap(norm(x)) for x in self.list_toptemps]
        tank_bottom_colors = [cmap(norm(x-self.params.delta_T(x))) for x in self.list_toptemps]
        list_thermoclines_reversed = [self.params.num_layers-x+1 for x in self.list_thermoclines]
        ax[1].bar(list_time, list_thermoclines_reversed, color=tank_bottom_colors, alpha=0.7)
        ax[1].bar(list_time, self.list_thermoclines, bottom=list_thermoclines_reversed, color=tank_top_colors, alpha=0.7)
        ax[1].set_xlabel('Time [hours]')
        ax[1].set_ylabel('Storage state')
        ax[1].set_ylim([0, self.params.num_layers])
        ax[1].set_yticks([])
        if len(list_time)>10 and len(list_time)<50:
            ax[1].set_xticks(list(range(0,len(list_time)+1,2)))
        ax3 = ax[1].twinx()
        ax3.plot(list_time, list_soc, color='black', alpha=0.4, label='SoC')
        ax3.set_ylabel('State of charge [%]')
        ax3.set_ylim([-1,101])
        # Color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.025, pad=0.15, alpha=0.7)
        cbar.set_label('Temperature [C]')


        # boundaries = np.arange(self.params.available_top_temps[0]*1.5,self.params.available_top_temps[-1]*1.5, 10)
        # colors = [plt.cm.Reds(i/(len(boundaries)-1)) for i in range(len(boundaries))]
        # cmap = ListedColormap(colors)
        # norm = BoundaryNorm(boundaries, cmap.N, clip=True)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.025, pad=0.15, alpha=0.7)
        # cbar.set_ticks(self.params.available_top_temps)
        # cbar.set_label('Temperature [F]')
        plt.savefig('plot.png', dpi=130)
        plt.show()


g = DGraph()