from multiprocessing import current_process
from typing import List
from pydantic import BaseModel

import pandas as pd
from scipy.stats import beta
from scipy.spatial.distance import jensenshannon

from src.change_detector.stats_and_distance.stats_and_distance import JSD_distance, iter_geo_mean_estimator, geo_mean
from src.change_detector.control_state import ControlState
from src.change_detector.monitor_method import MonitorMethod
from src.change_detector.process.process import Process
from src.change_detector.detect_response import DetectResponse

class PDFChangeDetector:
    def __init__(self, 
                a_memory:float=0.00, 
                z1:float=0.68, 
                z2:float=0.95, 
                z3:float=0.997, 
                reference_size:int=20,
                monitor_size:int=100,
                method:str="truncated",
                ) -> None:
        
        # alpha fading memory parameter and attributes
        self.a_memory:float = a_memory
        
        #quantile parameters
        self._z1 = z1
        self._z2 = z2
        self._z3 = z3
        
        self._reference_size = reference_size
        self._monitor_size = monitor_size
        self._method = MonitorMethod[method.upper()]
        
        self.current_process:Process|None = None
        self.candidate_process:Process|None = None
        
    def detect_change(self, new_pdf:pd.Series) -> DetectResponse:
        if self.current_process is None:
            self.current_process = Process(monitor_method=self._method, 
                                        monitor_size=self._monitor_size, 
                                        ref_PDF=new_pdf)
            print("current process run_order:", self.current_process.run_order,"\n")
            # return self.current_process.actual_state, self.current_process.estimated_alpha, self.current_process.estimated_beta, None, None, None, None
            return DetectResponse(self.current_process.actual_state, 
                self.current_process.estimated_alpha, 
                self.current_process.estimated_beta, 
                None, 
                None,
                None, 
                None)
            
        self.current_process.run_order += 1
        print("current process run_order:", self.current_process.run_order,"\n")
        
        self.current_process.update_alpha_fading_pdf(new_pdf)
        
        if self.current_process.run_order <= self._reference_size:
            self.current_process.update_reference_PDF()
            pdf_dist = JSD_distance(self.current_process.reference_PDF, self.current_process.alpha_fading_pdf)
            if self._method == MonitorMethod.TRUNCATED:
                self.current_process.update_beta_distribution_parameters(pdf_dist)
            elif self._method == MonitorMethod.WINDOW:
                self.current_process.update_distances_window(pdf_dist)
                # print("PDF window:", self.current_process.distances_window, len(self.current_process.distances_window))
            self.current_process.actual_state = ControlState.LEARNING
            
            # return self.current_process.actual_state, self.current_process.estimated_alpha, self.current_process.estimated_beta, None, None, None, None
            return DetectResponse(self.current_process.actual_state, 
                self.current_process.estimated_alpha, 
                self.current_process.estimated_beta, 
                None, 
                None,
                None, 
                None)
        
        
        else:
            pdf_dist = JSD_distance(self.current_process.reference_PDF, self.current_process.alpha_fading_pdf)
            # print("PDF distance:", pdf_dist, "current_order:", self._run_order)
            # print("JSD_distance:", jensenshannon(self._reference_PDF, self._alpha_fading_pdf, base=2)) # alternatively
            
            if self._method == MonitorMethod.TRUNCATED:
                self.current_process.update_beta_distribution_parameters(pdf_dist)
            elif self._method == MonitorMethod.WINDOW:
                self.current_process.update_distances_window(pdf_dist)
                # print("PDF window:", self.current_process.distances_window, len(self.current_process.distances_window))
                self.current_process.update_beta_distribution_parameters_window()
            
            u1, u2, u3 = map(lambda x: beta.ppf(x, 
                                                self.current_process.estimated_alpha, 
                                                self.current_process.estimated_beta), 
                            map(lambda x: (1+x)/2,
                                                [self._z1, 
                                                self._z2, 
                                                self._z3]))
            if not any([self.current_process.min_u1, self.current_process.min_u2, self.current_process.min_u3]) or u1 < self.current_process.min_u1:
                self.current_process.min_u1, self.current_process.min_u2, self.current_process.min_u3 = u1, u2, u3
            # print("current u1:", u1)
            # print("min u1:", self._min_u1, "  min u2:", self._min_u2, "  min u3:", self._min_u3)
                
            if u1 > self.current_process.min_u2: # monitoring a new process if dist distribution diverges such that u1 > min_u2
                if self.candidate_process is None: # create new process
                    self.candidate_process = Process(monitor_method=self._method,
                                                    monitor_size=self._monitor_size,
                                                    ref_PDF=new_pdf)
                    print("Warning: Started tracking candidate process.")
                else: # update the candidate process
                    self.candidate_process.run_order += 1
                    self.candidate_process.update_alpha_fading_pdf(new_pdf)
                    
                    if self.candidate_process.run_order <= self._reference_size:
                        self.candidate_process.update_reference_PDF()
                        pdf_dist = JSD_distance(self.candidate_process.reference_PDF, self.candidate_process.alpha_fading_pdf)
                        if self._method == MonitorMethod.TRUNCATED:
                            self.candidate_process.update_beta_distribution_parameters(pdf_dist)
                        elif self._method == MonitorMethod.WINDOW:
                            self.candidate_process.update_distances_window(pdf_dist)    
                        self.candidate_process.actual_state = ControlState.LEARNING    
            
            if u1 < self.current_process.min_u2:
                if self.candidate_process is not None: # Forget candidate process
                    self.candidate_process = None
                self.current_process.actual_state = ControlState.IN_CONTROL
                # return self.current_process.actual_state, self.current_process.estimated_alpha, self.current_process.estimated_beta, self.current_process.min_u1, self.current_process.min_u2, self.current_process.min_u3, u1
                # return DetectResponse(self.current_process.actual_state, 
                #                     self.current_process.estimated_alpha, 
                #                     self.current_process.estimated_beta, 
                #                     self.current_process.min_u1, 
                #                     self.current_process.min_u2,
                #                     self.current_process.min_u3, 
                #                     u1)
            elif u1 < self.current_process.min_u3:
                self.current_process.actual_state = ControlState.WARNING
                #return self.current_process.actual_state, self.current_process.estimated_alpha, self.current_process.estimated_beta, self.current_process.min_u1, self.current_process.min_u2, self.current_process.min_u3, u1
                # return DetectResponse(self.current_process.actual_state, 
                #     self.current_process.estimated_alpha, 
                #     self.current_process.estimated_beta, 
                #     self.current_process.min_u1, 
                #     self.current_process.min_u2,
                #     self.current_process.min_u3, 
                #     u1)
            else:
                #self.current_process.actual_state = ControlState.OUT_OF_CONTROL
                print("Out of control: Replacing current process with candidate process.")
                self.current_process = self.candidate_process
                self.candidate_process = None
                u1 = u2 = u3 = None
                self.current_process.actual_state = ControlState.OUT_OF_CONTROL
                #return self.current_process.actual_state, self.current_process.estimated_alpha, self.current_process.estimated_beta, self.current_process.min_u1, self.current_process.min_u2, self.current_process.min_u3, u1  
            
            return DetectResponse(self.current_process.actual_state, 
                self.current_process.estimated_alpha, 
                self.current_process.estimated_beta, 
                self.current_process.min_u1, 
                self.current_process.min_u2,
                self.current_process.min_u3, 
                u1)