from src.change_detector.control_state import ControlState
import pandas as pd
from scipy.stats import beta
from scipy.spatial.distance import jensenshannon
from src.change_detector.stats_and_distance.stats_and_distance import JSD_distance, iter_geo_mean_estimator

class PDFChangeDetector:
    def __init__(self, a_memory:float=0.5, z1=0.68, z2=0.95, z3=0.997) -> None:
        
        #quantile parameters
        self._z1 = z1
        self._z2 = z2
        self._z3 = z3
        
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
        
        
        
    def _update_alpha_fading_pdf(self, 
                            current_pdf:pd.Series
                            ) -> None:
        damped_hist = self._alpha_fading_pdf*self.a_memory
        self._alpha_fading_pdf = (damped_hist + current_pdf)/(1+self.a_memory)
        
    @property
    def alpha_fading_pdf(self) -> pd.Series|None:
        return self._alpha_fading_pdf
        
    def _update_beta_distribution_parameters(self,
                                new_dist: float, 
                                # order:int, 
                                # old_dist_geo_mean:float, 
                                # old_complementary_dist_geo_mean:float
                                ) -> None:
        
        if self._run_order == 1:
            self._dist_geo_mean, self._complementary_dist_geo_mean = new_dist, 1-new_dist
        else:    
            self._update_geo_mean_estimators(new_dist)
            self._estimated_alpha = 1/2*(1+self._dist_geo_mean/(1-self._dist_geo_mean-self._complementary_dist_geo_mean))
            self._estimated_beta =  1/2*(1+self._complementary_dist_geo_mean/(1-self._dist_geo_mean-self._complementary_dist_geo_mean))
        print("new alpha:", self._estimated_alpha, "  new beta:", self._estimated_beta)
        print("geo_mean:", self._dist_geo_mean, "  complementary geo_mean:", self._complementary_dist_geo_mean)
            
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
        
    def _update_geo_mean_estimators(self, new_dist: float) ->None:
        self._dist_geo_mean, self._complementary_dist_geo_mean = (
            iter_geo_mean_estimator(x1, self._run_order, x2) for x1, x2 in
            [(new_dist, self._dist_geo_mean), (1-new_dist, self._complementary_dist_geo_mean)]
        )
    
    def detect_change(self,new_pdf:pd.Series) -> ControlState:
        
        if self._reference_PDF is None:
            self._reset_detector_parameters(new_pdf)
            print("First pdf taken as reference.")
            return ControlState.IN_CONTROL
        else:
            self._update_alpha_fading_pdf(new_pdf)
            pdf_dist = JSD_distance(self._reference_PDF, self._alpha_fading_pdf)
            print("PDF distance:", pdf_dist, "current_order:", self._run_order)
            # print("JSD_distance:", jensenshannon(self._reference_PDF, self._alpha_fading_pdf, base=2)) # alternatively
            self._update_beta_distribution_parameters(pdf_dist)
            self._run_order += 1
            if not all([self._estimated_alpha, self._estimated_beta]):
                print("Only one distance estimator is available.")
                return ControlState.IN_CONTROL
            u1, u2, u3 = map(lambda x: beta.ppf(x, self._estimated_alpha, self._estimated_beta), 
                            map(lambda x: (1+x)/2,
                                [self._z1, self._z2, self._z3]))
            if not any([self._min_u1, self._min_u2, self._min_u3]) or u1 < self._min_u1:
                self._min_u1, self._min_u2, self._min_u3 = u1, u2, u3
            print("current u1:", u1)
            print("min u1:", self._min_u1, "  min u2:", self._min_u2, "  min u3:", self._min_u3)
            if u1 < self._min_u2:
                return ControlState.IN_CONTROL
            elif u1 < self._min_u3:
                return ControlState.WARNING
            else:
                self._reset_detector_parameters(new_pdf)
                print("Out of control. New pdf taken as reference.")
                return ControlState.OUT_OF_CONTROL
        