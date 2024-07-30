from typing import List

import pandas as pd
from scipy.stats import beta
from scipy.spatial.distance import jensenshannon

from src.change_detector.stats_and_distance.stats_and_distance import JSD_distance, iter_geo_mean_estimator, geo_mean
from src.change_detector.control_state import ControlState

class PDFChangeDetector:
    def __init__(self, 
                a_memory:float=0.00, 
                z1:float=0.68, 
                z2:float=0.95, 
                z3:float=0.997, 
                reference_size:int=20,
                monitor_size:int=100,
                method:str='truncated',
                ) -> None:
        
        #quantile parameters
        self._z1 = z1
        self._z2 = z2
        self._z3 = z3
        
        self._reference_size = reference_size
        self._monitor_size = monitor_size
        self._method = method
        
        # alpha fading memory parameter and attributes
        self.a_memory:float = a_memory
        self._alpha_fading_pdf:pd.Series|None = None
        
        # Beta distribution parameters
        self._estimated_alpha:float|None = None
        self._estimated_beta:float|None = None
        
        # Stochastic process parameters and attributes
        self._run_order = 1
        
        self._reference_PDF = None
        
        self._dist_geo_mean:float|None = None
        self._complementary_dist_geo_mean:float|None = None
        
        self._min_u1 = None
        self._min_u2 = None
        self._min_u3 = None
        
        # warning update memory
        self._new_run_order = 0
        
        
        if self._method == "window":
            self._distances_window = []
        
        
    def _update_alpha_fading_pdf(self, 
                            current_pdf:pd.Series
                            ) -> None:
        damped_hist = self._alpha_fading_pdf*self.a_memory
        self._alpha_fading_pdf = (damped_hist + current_pdf)/(1+self.a_memory)
        
    @property
    def alpha_fading_pdf(self) -> pd.Series|None:
        return self._alpha_fading_pdf
    
    #********************************
    def _update_geo_mean_estimators(self, new_dist: float):# dist_geo_mean, complementary_dist_geo_mean, run_order:int = 1, order_limit:int|None = None) -> (float, float):
        self._dist_geo_mean, self._complementary_dist_geo_mean = (
            iter_geo_mean_estimator(x1, self._run_order, x2, order_limit=self._monitor_size) for x1, x2 in
            [(new_dist, self._dist_geo_mean), (1-new_dist, self._complementary_dist_geo_mean)]
        )
        # return dist_geo_mean, complementary_dist_geo_mean

    def _update_beta_distribution_parameters(
                            self,
                            new_dist: float, 
                            # run_order:int=self, 
                            # old_dist_geo_mean:float, 
                            # old_complementary_dist_geo_mean:float,
                            # order_limit:int|None = None
                            ) -> None:
    
        if self._run_order == 2:
            self._dist_geo_mean, self._complementary_dist_geo_mean = new_dist, 1-new_dist
        else:    
            self._update_geo_mean_estimators(new_dist)# old_dist_geo_mean, old_complementary_dist_geo_mean, run_order, order_limit)
            self._estimated_alpha = 1/2+self._dist_geo_mean/(2*(1-self._dist_geo_mean-self._complementary_dist_geo_mean))
            self._estimated_beta =  1/2+self._complementary_dist_geo_mean/(2*(1-self._dist_geo_mean-self._complementary_dist_geo_mean))
            # print("new alpha:", self._estimated_alpha, "  new beta:", self._estimated_beta)
            # print("geo_mean:", self._dist_geo_mean, "  complementary geo_mean:", self._complementary_dist_geo_mean)
    
    def _update_beta_distribution_parameters_window(self,
                            distances_window:List[float]
                            ) -> None:
        self._dist_geo_mean, self._complementary_dist_geo_mean = geo_mean(distances_window), geo_mean([1-dist for dist in distances_window])
        self._estimated_alpha = 1/2+self._dist_geo_mean/(2*(1-self._dist_geo_mean-self._complementary_dist_geo_mean))
        self._estimated_beta =  1/2+self._complementary_dist_geo_mean/(2*(1-self._dist_geo_mean-self._complementary_dist_geo_mean))
        # print("new alpha:", self._estimated_alpha, "  new beta:", self._estimated_beta)

    def _reset_detector_parameters(self, ref_PDF:pd.Series) -> None:
        self._estimated_alpha = None
        self._estimated_beta = None
        self._dist_geo_mean = None
        self._complementary_dist_geo_mean = None
        self._run_order = 1
        self._alpha_fading_pdf = None
        self._reference_PDF = ref_PDF
        self._alpha_fading_pdf = ref_PDF
        self._min_u1, self._min_u2, self._min_u3 = None, None, None
        if self._method == "window":
            self._distances_window = []
        
    def detect_change(self,new_pdf:pd.Series) -> tuple[ControlState, float|None, float|None, float|None, float|None, float|None, float|None]:
        
        if self._reference_PDF is None:
            print("run_order:", self._run_order,"\n")
            self._reset_detector_parameters(new_pdf)
            return ControlState.LEARNING, self._estimated_alpha, self._estimated_beta, None, None, None, None
        
        self._run_order += 1
        print("\n***\nrun_order:", self._run_order,"\n")
        if self._run_order <= self._reference_size:
            
            self._update_alpha_fading_pdf(new_pdf)
            pdf_dist = JSD_distance(self._reference_PDF, self._alpha_fading_pdf)
            if self._method == "truncated":
                self._update_beta_distribution_parameters(pdf_dist)
            elif self._method == "window":
                self._distances_window.append(pdf_dist)
            self._reference_PDF = (self._reference_PDF*(self._run_order-1)+self._alpha_fading_pdf)/self._run_order
            return ControlState.LEARNING, self._estimated_alpha, self._estimated_beta, None, None, None, None
        
        else:
            self._update_alpha_fading_pdf(new_pdf)
            pdf_dist = JSD_distance(self._reference_PDF, self._alpha_fading_pdf)
            # print("PDF distance:", pdf_dist, "current_order:", self._run_order)
            # print("JSD_distance:", jensenshannon(self._reference_PDF, self._alpha_fading_pdf, base=2)) # alternatively
            
            if self._method == "truncated":
                self._update_beta_distribution_parameters(pdf_dist)
            elif self._method == "window":
                self._distances_window.append(pdf_dist)
                if len(self._distances_window) > self._monitor_size:
                    self._distances_window.pop(0)
                print("PDF window:", self._distances_window, len(self._distances_window))
                self._update_beta_distribution_parameters_window(self._distances_window)
            
            
            # if not all([self._estimated_alpha, self._estimated_beta]):
            #     # print("Only one distance estimator is available.")
            #     return ControlState.IN_CONTROL, self._estimated_alpha, self._estimated_beta
            u1, u2, u3 = map(lambda x: beta.ppf(x, self._estimated_alpha, self._estimated_beta), 
                            map(lambda x: (1+x)/2,
                                [self._z1, self._z2, self._z3]))
            if not any([self._min_u1, self._min_u2, self._min_u3]) or u1 < self._min_u1:
                self._min_u1, self._min_u2, self._min_u3 = u1, u2, u3
            # print("current u1:", u1)
            # print("min u1:", self._min_u1, "  min u2:", self._min_u2, "  min u3:", self._min_u3)
            if u1 < self._min_u2:
                return ControlState.IN_CONTROL, self._estimated_alpha, self._estimated_beta, self._min_u1, self._min_u2, self._min_u3, u1
            elif u1 < self._min_u3:
                
                return ControlState.WARNING, self._estimated_alpha, self._estimated_beta, self._min_u1, self._min_u2, self._min_u3, u1
            else:
                self._reset_detector_parameters(new_pdf)
                # print("Out of control. New pdf taken as reference.")
                return ControlState.OUT_OF_CONTROL, self._estimated_alpha, self._estimated_beta, self._min_u1, self._min_u2, self._min_u3, u1
        