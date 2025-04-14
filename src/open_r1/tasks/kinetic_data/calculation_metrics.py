import numpy as np
from scipy.stats import linregress
from typing import Dict, Any
from pathlib import Path
import sys


class KineticMetricsCalculator:
    def __init__(self, kinetic_data: Dict[str, Any]):
        self.data = kinetic_data

    def calculate_all_metrics(self):
        metrics = {}
        for run_id, run_data in self.data.items():
            metrics[run_id] = {
                "reaction_order": self._calc_reaction_order(run_data),
                "TOF": self._calc_tof(run_data),
                "TON": self._calc_ton(run_data),
                "catalyst_stability": self._calc_stability(run_data),
                "induction_period": self._calc_induction_period(run_data),
                "mass_balance_gap": self._calc_mass_balance_gap(run_data),
                "deactivation_rate_constant": self._calc_deactivation(run_data),
                "time_max_curvature": self._calc_time_max_curvature(run_data),
                "active_catalyst_fraction": self._calc_active_catalyst_fraction(run_data),
                "catalyst_activity_half_life": self._calc_activity_half_life(run_data),
                "Keq": self._calc_Keq(run_data),
                "SP_mid_ratio": self._calc_SP_mid_ratio(run_data),
                "mass_gap_mid": self._calc_mass_gap_mid(run_data)
            }
        return metrics

    def summarize_metrics_for_ml(self) -> Dict[str, float]:
        metrics = self.calculate_all_metrics()
        numeric_metrics = ["TOF", "TON", "catalyst_stability", "induction_period",
                           "mass_balance_gap", "deactivation_rate_constant",
                           "time_max_curvature", "active_catalyst_fraction",
                           "catalyst_activity_half_life", "Keq",
                           "SP_mid_ratio", "mass_gap_mid"]

        summary = {}
        for metric in numeric_metrics:
            values = np.array([metrics[run][metric] for run in metrics])
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)

        order_mapping = {"zero-order": 0, "first-order": 1, "second-order": 2}
        orders = np.array([order_mapping[metrics[run]["reaction_order"]] for run in metrics])
        summary["reaction_order_mean"] = np.mean(orders)
        summary["reaction_order_std"] = np.std(orders)

        return summary

    def _calc_reaction_order(self, run_data):
        t = np.array(run_data["time_data"])
        s = np.array(run_data["substrate_data"])
        
        zero_order_fit = linregress(t, s)
        first_order_fit = linregress(t, np.log(np.clip(s, 1e-8, None)))
        second_order_fit = linregress(t, 1/np.clip(s, 1e-8, None))
        
        fits = [abs(zero_order_fit.rvalue), abs(first_order_fit.rvalue), abs(second_order_fit.rvalue)]
        order = np.argmax(fits)
        
        return ["zero-order", "first-order", "second-order"][order]

    def _calc_tof(self, run_data):
        cat_conc = run_data["initial_concentration_of_catalyst"]
        t = np.array(run_data["time_data"][:3])
        p = np.array(run_data["product_data"][:3])
        slope, _, _, _, _ = linregress(t, p)
        return slope / cat_conc

    def _calc_ton(self, run_data):
        cat_conc = run_data["initial_concentration_of_catalyst"]
        total_product = run_data["product_data"][-1] - run_data["product_data"][0]
        return total_product / cat_conc

    def _calc_stability(self, run_data):
        t = np.array(run_data["time_data"])
        p = np.array(run_data["product_data"])
        rate_initial = (p[1] - p[0]) / (t[1] - t[0])
        rate_final = (p[-1] - p[-2]) / (t[-1] - t[-2])
        stability = rate_final / rate_initial
        return stability  # Stability ≈1 (stable), <<1 (unstable)

    def _calc_induction_period(self, run_data):
        p = np.array(run_data["product_data"])
        t = np.array(run_data["time_data"])
        threshold = 0.05 * (p[-1] - p[0])  # 5% of total product
        indices_above_thresh = np.where(p - p[0] > threshold)[0]
        induction_time = t[indices_above_thresh[0]] if len(indices_above_thresh) > 0 else 0
        return induction_time

    def _calc_mass_balance_gap(self, run_data):
        s0 = run_data["substrate_data"][0]
        s = np.array(run_data["substrate_data"])
        p = np.array(run_data["product_data"])
        gap = np.abs((s0 - s[-1]) - (p[-1] - p[0]))
        return gap

    def _calc_deactivation(self, run_data):
        t = np.array(run_data["time_data"])
        p = np.array(run_data["product_data"])
        rates = np.diff(p) / np.diff(t)
        t_mid = (t[:-1] + t[1:]) / 2
        ln_rates = np.log(np.clip(rates, 1e-8, None))
        slope, _, _, _, _ = linregress(t_mid, ln_rates)
        return -slope

    def _calc_time_max_curvature(self, run_data):
        t = np.array(run_data["time_data"])
        p = np.array(run_data["product_data"])
        d2p_dt2 = np.gradient(np.gradient(p, t), t)
        return t[np.argmax(d2p_dt2)]

    def _calc_active_catalyst_fraction(self, run_data):
        t = np.array(run_data["time_data"])
        p = np.array(run_data["product_data"])
        initial_rate = (p[1] - p[0]) / (t[1] - t[0])
        final_rate = (p[-1] - p[-2]) / (t[-1] - t[-2])
        return final_rate / initial_rate if initial_rate > 1e-8 else 0

    def _calc_activity_half_life(self, run_data):
        t = np.array(run_data["time_data"])
        p = np.array(run_data["product_data"])
        rates = np.diff(p) / np.diff(t)
        half_initial_rate = rates[0] / 2
        idx_half_life = np.where(rates <= half_initial_rate)[0]
        return t[idx_half_life[0]] if len(idx_half_life) > 0 else t[-1]

    def _calc_Keq(self, run_data):
        substrate_final = run_data["substrate_data"][-1]
        product_final = run_data["product_data"][-1]
        return product_final / substrate_final if substrate_final > 1e-6 else np.inf

    def _calc_SP_mid_ratio(self, run_data):
        midpoint = len(run_data["time_data"]) // 2
        substrate_mid = run_data["substrate_data"][midpoint]
        product_mid = run_data["product_data"][midpoint]
        return substrate_mid / product_mid if product_mid > 1e-6 else np.inf

    def _calc_mass_gap_mid(self, run_data):
        midpoint = len(run_data["time_data"]) // 2
        s0 = run_data["substrate_data"][0]
        s_mid = run_data["substrate_data"][midpoint]
        p_mid = run_data["product_data"][midpoint]
        p0 = run_data["product_data"][0]
        return abs((s0 - s_mid) - (p_mid - p0))
