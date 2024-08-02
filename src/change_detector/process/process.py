from dataclasses import dataclass, field, InitVar
from types import new_class
from typing import List

import pandas as pd

from src.change_detector.stats_and_distance.stats_and_distance import JSD_distance, iter_geo_mean_estimator, geo_mean
from src.change_detector.control_state import ControlState
from src.change_detector.monitor_method import MonitorMethod


class Process():
    def __init__(self,
                monitor_method:MonitorMethod,
                monitor_size:int,
                ref_PDF:pd.Series|None = None,
                ):
        self._monitor_method = monitor_method
        self._monitor_size = monitor_size
        
        self.reference_PDF:pd.Series|None = ref_PDF
        self.alpha_fading_pdf:pd.Series|None = ref_PDF # replaceable with current_PDF
    
        # Beta distribution parameters
        self.estimated_alpha:float|None = None
        self.estimated_beta:float|None = None
        
        # Stochastic process parameters and attributes
        self.run_order:int = 1
        
        self.dist_geo_mean:float|None = None
        self.complementary_dist_geo_mean:float|None = None
        
        self.min_u1:float|None = None
        self.min_u2:float|None = None
        self.min_u3:float|None = None
        
        self.actual_state:ControlState = ControlState.LEARNING

        if monitor_method == MonitorMethod.WINDOW:
            self.distances_window:List = []
            
    def update_alpha_fading_pdf(self, # replaceable with current_PDF
                            current_pdf:pd.Series,
                            a_memory:float=0.0
                            ) -> None:

        damped_hist = self.alpha_fading_pdf*a_memory
        self.alpha_fading_pdf = (damped_hist + current_pdf)/(1+a_memory)
        
    def update_reference_PDF(self, 
                            new_pdf:pd.Series|None = None,
                            ) -> None:
        if new_pdf is None:
            new_pdf=self.alpha_fading_pdf
        self.reference_PDF = (self.reference_PDF*(self.run_order-1)+new_pdf)/self.run_order

    #********************************
    def _update_iter_geo_mean_estimators(self, 
                                    new_dist: float,
                                    ):
        
        self.dist_geo_mean, self.complementary_dist_geo_mean = (
            iter_geo_mean_estimator(x1, self.run_order, x2, order_limit=self._monitor_size) for x1, x2 in
            [(new_dist, self.dist_geo_mean), (1-new_dist,self.complementary_dist_geo_mean)]
        )

    def update_beta_distribution_parameters(
                            self,
                            new_dist: float,
                            ) -> None:

        if self.run_order == 2:
            self.dist_geo_mean, self.complementary_dist_geo_mean = new_dist, 1-new_dist
        else:    
            self._update_iter_geo_mean_estimators(new_dist)# old_dist_geo_mean, old_complementary_dist_geo_mean, run_order, order_limit)
            self.estimated_alpha = 1/2+self.dist_geo_mean/(2*(1-self.dist_geo_mean-self.complementary_dist_geo_mean))
            self.estimated_beta =  1/2+self.complementary_dist_geo_mean/(2*(1-self.dist_geo_mean-self.complementary_dist_geo_mean))
            # print("new alpha:", self._estimated_alpha, "  new beta:", self._estimated_beta)
            # print("geo_mean:", self._dist_geo_mean, "  complementary geo_mean:", self._complementary_dist_geo_mean)
    
    def update_distances_window(self, 
                                new_distance:float,
                                ) -> None:
        self.distances_window.append(new_distance)
        self.distances_window = self.distances_window[-self._monitor_size:]
        
    def update_beta_distribution_parameters_window(self,
                            distances_window:List[float]|None = None,
                            ) -> None:
        if distances_window is None:
            distances_window = self.distances_window
        self.dist_geo_mean, self.complementary_dist_geo_mean = geo_mean(distances_window), geo_mean([1-dist for dist in distances_window])
        self.estimated_alpha = 1/2+self.dist_geo_mean/(2*(1-self.dist_geo_mean-self.complementary_dist_geo_mean))
        self.estimated_beta =  1/2+self.complementary_dist_geo_mean/(2*(1-self.dist_geo_mean-self.complementary_dist_geo_mean))
        # print("new alpha:", self._estimated_alpha, "  new beta:", self._estimated_beta)

    # def _reset_detector_parameters(self, ref_PDF:pd.Series) -> None:
    #     self._estimated_alpha = None
    #     self._estimated_beta = None
    #     self._dist_geo_mean = None
    #     self._complementary_dist_geo_mean = None
    #     self._run_order = 1
    #     self._alpha_fading_pdf = None
    #     self._reference_PDF = ref_PDF
    #     self._alpha_fading_pdf = ref_PDF
    #     self._min_u1, self._min_u2, self._min_u3 = None, None, None
    #     self._actual_state = ControlState.LEARNING
    #     if self._method == "window":
    #         self._distances_window = []