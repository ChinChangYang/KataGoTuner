from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
import math
import numpy as np
import scipy
import scipy.optimize
import random
import time

def _uniform_strictly_between(a: float, b:float) -> float:
    if b <= a:
        return (a+b) / 2.0
    while True:
        r = random.uniform(a,b)
        if r > a and r < b:
            return r

def logprob_of_win(lgamma_advantage: float) -> float:
    if lgamma_advantage < -40.0:
        return lgamma_advantage
    return -np.log1p(np.exp(-lgamma_advantage))

def logprob_of_win_first_derivative(lgamma_advantage: float) -> float:
    return 1.0 / (1.0 + np.exp(lgamma_advantage))

def logprob_of_win_second_derivative(lgamma_advantage: float) -> float:
    return -1.0 / (np.exp(-lgamma_advantage/2.0) + np.exp(lgamma_advantage/2.0)) ** 2.0

def logistic_negative_log_prob(
    coeffs: np.ndarray,
    winlossdiff_weight_datas: List[Tuple[np.ndarray,float]],
    additional_weight_factors: np.ndarray,
    num_linear_features: int,
    num_quadratic_features: int,
    linear_coeff_prior_stdev: float,
    quadratic_coeff_prior_stdev: float,
) -> float:
    accum_logprob = 0.0
    assert len(winlossdiff_weight_datas) == len(additional_weight_factors)
    for i in range(len(winlossdiff_weight_datas)):
        winnerloser_features_diff, weight = winlossdiff_weight_datas[i]
        additional_weight_factor = additional_weight_factors[i]
        lgamma_advantage = np.dot(winnerloser_features_diff, coeffs)
        accum_logprob += additional_weight_factor * weight * logprob_of_win(lgamma_advantage)

    assert len(coeffs) == num_linear_features + num_quadratic_features
    accum_logprob -= 0.5 * np.dot(coeffs[:num_linear_features],coeffs[:num_linear_features]) / linear_coeff_prior_stdev / linear_coeff_prior_stdev
    accum_logprob -= 0.5 * np.dot(coeffs[num_linear_features:], coeffs[num_linear_features:]) / quadratic_coeff_prior_stdev / quadratic_coeff_prior_stdev
    # print("coeffs", coeffs, "accum_logprob", accum_logprob)
    return -accum_logprob

def logistic_negative_log_prob_derivative(
    coeffs: np.ndarray,
    winlossdiff_weight_datas: List[Tuple[np.ndarray,float]],
    additional_weight_factors: np.ndarray,
    num_linear_features: int,
    num_quadratic_features: int,
    linear_coeff_prior_stdev: float,
    quadratic_coeff_prior_stdev: float,
) -> np.ndarray:
    accum_dlogprob_dcoeff = np.zeros(len(coeffs))
    assert len(winlossdiff_weight_datas) == len(additional_weight_factors)
    for i in range(len(winlossdiff_weight_datas)):
        winnerloser_features_diff, weight = winlossdiff_weight_datas[i]
        additional_weight_factor = additional_weight_factors[i]
        lgamma_advantage = np.dot(winnerloser_features_diff, coeffs)
        accum_dlogprob_dcoeff += (additional_weight_factor * weight * logprob_of_win_first_derivative(lgamma_advantage)) * winnerloser_features_diff

    assert len(coeffs) == num_linear_features + num_quadratic_features
    accum_dlogprob_dcoeff[:num_linear_features] -= coeffs[:num_linear_features] / linear_coeff_prior_stdev / linear_coeff_prior_stdev
    accum_dlogprob_dcoeff[num_linear_features:] -= coeffs[num_linear_features:] / quadratic_coeff_prior_stdev / quadratic_coeff_prior_stdev
    # print("coeffs", coeffs, "accum_logprob", accum_logprob)
    return -accum_dlogprob_dcoeff

def logistic_negative_log_prob_second_derivative(
    coeffs: np.ndarray,
    winlossdiff_weight_datas: List[Tuple[np.ndarray,float]],
    additional_weight_factors: np.ndarray,
    num_linear_features: int,
    num_quadratic_features: int,
    linear_coeff_prior_stdev: float,
    quadratic_coeff_prior_stdev: float,
) -> np.ndarray:
    accum_d2logprob_dcoeff2 = np.zeros((len(coeffs),len(coeffs)))
    assert len(winlossdiff_weight_datas) == len(additional_weight_factors)
    for i in range(len(winlossdiff_weight_datas)):
        winnerloser_features_diff, weight = winlossdiff_weight_datas[i]
        additional_weight_factor = additional_weight_factors[i]
        lgamma_advantage = np.dot(winnerloser_features_diff, coeffs)
        accum_d2logprob_dcoeff2 += (additional_weight_factor * weight * logprob_of_win_second_derivative(lgamma_advantage)) * np.outer(winnerloser_features_diff, winnerloser_features_diff)

    assert len(coeffs) == num_linear_features + num_quadratic_features
    for i in range(num_linear_features):
        accum_d2logprob_dcoeff2[i,i] -= 1.0 / linear_coeff_prior_stdev / linear_coeff_prior_stdev
    for i in range(num_linear_features,len(coeffs)):
        accum_d2logprob_dcoeff2[i,i] -= 1.0 / quadratic_coeff_prior_stdev / quadratic_coeff_prior_stdev
    # print("coeffs", coeffs, "accum_logprob", accum_logprob)
    return -accum_d2logprob_dcoeff2

def quadratic_regress(
    win_loss_weight_datas: List[Tuple[np.ndarray,np.ndarray,float]],
    additional_weight_factors: np.ndarray,
    initial_linear_coeffs: np.ndarray,
    initial_quadratic_coeffs: np.ndarray,
    num_features: int,
    linear_coeff_prior_stdev: float,
    quadratic_coeff_prior_stdev: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Add quadratic regressors and pre-difference the resulting vectors
    winlossdiff_weight_datas = []
    for winner_features, loser_features, weight in win_loss_weight_datas:
        assert len(winner_features) == num_features
        assert len(loser_features) == num_features
        new_winner_features = np.append(winner_features, np.outer(winner_features, winner_features).flatten())
        new_loser_features = np.append(loser_features, np.outer(loser_features, loser_features).flatten())
        winlossdiff_weight_datas.append((new_winner_features-new_loser_features, weight))

    result = scipy.optimize.minimize(
        fun=logistic_negative_log_prob,
        x0=np.append(initial_linear_coeffs, initial_quadratic_coeffs),
        args=(winlossdiff_weight_datas,
         additional_weight_factors,
         num_features,
         num_features*num_features,
         linear_coeff_prior_stdev,
         quadratic_coeff_prior_stdev,
        ),
        options=dict(disp=False,maxiter=2),
        method='trust-exact',
        jac=logistic_negative_log_prob_derivative,
        hess=logistic_negative_log_prob_second_derivative,
    )
    linear_coeffs = result.x[:num_features]
    quadratic_coeffs = result.x[num_features:]
    return linear_coeffs, quadratic_coeffs

def logistic_mean_and_confidence_deviation(
    win_loss_weight_datas: List[Tuple[np.ndarray,np.ndarray,float]],
    additional_winner_weight_factors: np.ndarray,
    additional_loser_weight_factors: np.ndarray,
    linear_coeffs: np.ndarray,
    quadratic_coeffs: np.ndarray,
    constant_coeff_prior_stdev: float,
) -> Tuple[float,float]:
    # t0 = time.time()
    winlgamma_losslgamma_weight_datas = []
    for i in range(len(win_loss_weight_datas)):
        winner_features, loser_features, weight = win_loss_weight_datas[i]
        winner_lgamma = np.dot(winner_features, linear_coeffs) + np.dot(np.outer(winner_features,winner_features).flatten(), quadratic_coeffs)
        loser_lgamma = np.dot(loser_features, linear_coeffs) + np.dot(np.outer(loser_features,loser_features).flatten(), quadratic_coeffs)
        winlgamma_losslgamma_weight_datas.append((
            winner_lgamma,
            loser_lgamma,
            additional_winner_weight_factors[i] * weight,
            additional_loser_weight_factors[i] * weight
        ))

    # Compute the weighted arithmetic mean of the loggamma of the players.
    def compute_arithmetic_lgamma_mean():
        wxsum = 0.0
        wsum = 0.0
        for i in range(len(winlgamma_losslgamma_weight_datas)):
            winner_lgamma, loser_lgamma, winner_weight, loser_weight = winlgamma_losslgamma_weight_datas[i]
            wxsum += 0.5 * winner_weight * winner_lgamma
            wsum += 0.5 * winner_weight
            wxsum += 0.5 * loser_weight * loser_lgamma
            wsum += 0.5 * loser_weight
        return wxsum / (1e-30 + wsum)

    arithmetic_lgamma_mean = compute_arithmetic_lgamma_mean()
    # t1 = time.time()

    # Optimize logistic_mean to be the believed strength of a hypothetical player who
    # drew games against each player in each datapoint according to that data point's weight
    # def global_mean_player_neg_logprob(coeffs: np.ndarray):
    #     assert len(coeffs) == 1
    #     logistic_mean_lgamma = coeffs[0]
    #     accum_logprob = 0.0
    #     assert len(win_loss_weight_datas) == len(additional_winner_weight_factors)
    #     assert len(win_loss_weight_datas) == len(additional_loser_weight_factors)
    #     for i in range(len(winlgamma_losslgamma_weight_datas)):
    #         winner_lgamma, loser_lgamma, winner_weight, loser_weight = winlgamma_losslgamma_weight_datas[i]
    #         accum_logprob += winner_weight * 0.25 * logprob_of_win(logistic_mean_lgamma - winner_lgamma)
    #         accum_logprob += winner_weight * 0.25 * logprob_of_win(winner_lgamma - logistic_mean_lgamma)
    #         accum_logprob += loser_weight * 0.25 * logprob_of_win(logistic_mean_lgamma - loser_lgamma)
    #         accum_logprob += loser_weight * 0.25 * logprob_of_win(loser_lgamma - logistic_mean_lgamma)

    #     # Add a tiny force regularizing logistic_mean_lgamma to be the arithmetic mean
    #     accum_logprob -= 0.5 * (logistic_mean_lgamma - arithmetic_lgamma_mean) ** 2.0 / constant_coeff_prior_stdev / constant_coeff_prior_stdev
    #     return -accum_logprob

    # def global_mean_player_neg_logprob_first_derivative(coeffs: np.ndarray):
    #     assert len(coeffs) == 1
    #     logistic_mean_lgamma = coeffs[0]
    #     accum_dlogprob_dcoeff = 0.0
    #     assert len(win_loss_weight_datas) == len(additional_winner_weight_factors)
    #     assert len(win_loss_weight_datas) == len(additional_loser_weight_factors)
    #     for i in range(len(winlgamma_losslgamma_weight_datas)):
    #         winner_lgamma, loser_lgamma, winner_weight, loser_weight = winlgamma_losslgamma_weight_datas[i]
    #         accum_dlogprob_dcoeff += winner_weight * 0.25 * logprob_of_win_first_derivative(logistic_mean_lgamma - winner_lgamma)
    #         accum_dlogprob_dcoeff += -winner_weight * 0.25 * logprob_of_win_first_derivative(winner_lgamma - logistic_mean_lgamma)
    #         accum_dlogprob_dcoeff += loser_weight * 0.25 * logprob_of_win_first_derivative(logistic_mean_lgamma - loser_lgamma)
    #         accum_dlogprob_dcoeff += -loser_weight * 0.25 * logprob_of_win_first_derivative(loser_lgamma - logistic_mean_lgamma)

    #     # Add a tiny force regularizing logistic_mean_lgamma to be the arithmetic mean
    #     accum_dlogprob_dcoeff -= (logistic_mean_lgamma - arithmetic_lgamma_mean) / constant_coeff_prior_stdev / constant_coeff_prior_stdev
    #     return -accum_dlogprob_dcoeff

    # The second derivative of global_mean_player_neg_logprob
    # This is the local precision of the belief about the hypothetical player's loggamma
    def global_mean_player_neg_logprob_second_derivative(coeffs: np.ndarray):
        assert len(coeffs) == 1
        logistic_mean_lgamma = coeffs[0]
        accum_neg_precision = 0.0
        assert len(win_loss_weight_datas) == len(additional_winner_weight_factors)
        assert len(win_loss_weight_datas) == len(additional_loser_weight_factors)
        for i in range(len(win_loss_weight_datas)):
            winner_lgamma, loser_lgamma, winner_weight, loser_weight = winlgamma_losslgamma_weight_datas[i]
            accum_neg_precision += winner_weight * 0.25 * logprob_of_win_second_derivative(logistic_mean_lgamma - winner_lgamma)
            accum_neg_precision += winner_weight * 0.25 * logprob_of_win_second_derivative(winner_lgamma - logistic_mean_lgamma)
            accum_neg_precision += loser_weight * 0.25 * logprob_of_win_second_derivative(logistic_mean_lgamma - loser_lgamma)
            accum_neg_precision += loser_weight * 0.25 * logprob_of_win_second_derivative(loser_lgamma - logistic_mean_lgamma)

        # Add a tiny force regularizing logistic_mean_lgamma to be the arithmetic mean
        accum_neg_precision -= 1.0 / constant_coeff_prior_stdev / constant_coeff_prior_stdev
        return np.array([[-accum_neg_precision]])

    # result = scipy.optimize.minimize(
    #     fun=global_mean_player_neg_logprob,
    #     x0=np.array([arithmetic_lgamma_mean]),
    #     args=tuple(),
    #     options=dict(disp=False),
    #     method='trust-exact',
    #     jac=global_mean_player_neg_logprob_first_derivative,
    #     hess=global_mean_player_neg_logprob_second_derivative,
    # )
    # t2 = time.time()
    # logistic_mean_lgamma = result.x[0]

    logistic_mean_lgamma = arithmetic_lgamma_mean

    precision = global_mean_player_neg_logprob_second_derivative(np.array([logistic_mean_lgamma]))[0,0]
    # t3 = time.time()
    # print(t1-t0,t2-t1,t3-t2,logistic_mean_lgamma-arithmetic_lgamma_mean)
    return logistic_mean_lgamma, math.sqrt(1.0 / precision)

@dataclass
class Parameter:
    name: str
    guess: float
    likely_radius_around_guess: float
    hard_lower_bound: Optional[float]
    hard_upper_bound: Optional[float]

class PairwiseCLOP:
    def __init__(
        self,
        parameters: List[Parameter],
        constant_coeff_prior_stdev: float = 10.0,
        linear_coeff_prior_stdev: float = 10.0,
        quadratic_coeff_prior_stdev: float = 10.0,
        weight_factor_shrink_stop_threshold: float = 0.001,
        max_iters: int = 1000,
        locality_stdevs: float = 3.0,
        num_gibbs_sampling_rounds: int = 20,
        interp_to_use_max_pair_weight: float = 0.00,
        debug: bool = False,
    ):
        self.num_parameters = len(parameters)
        self.parameters = parameters
        self.constant_coeff_prior_stdev = constant_coeff_prior_stdev
        self.linear_coeff_prior_stdev = linear_coeff_prior_stdev
        self.quadratic_coeff_prior_stdev = quadratic_coeff_prior_stdev
        self.weight_factor_shrink_stop_threshold = weight_factor_shrink_stop_threshold
        self.max_iters = max_iters
        self.locality_stdevs = locality_stdevs
        self.num_gibbs_sampling_rounds = num_gibbs_sampling_rounds
        self.interp_to_use_max_pair_weight = interp_to_use_max_pair_weight
        self.debug = debug

        self.observations: List[Tuple[np.ndarray,np.ndarray,float]] = []
        self.current_optimum: np.ndarray = np.zeros(self.num_parameters)
        self.current_quadratic_fits: List[Tuple[np.ndarray,np.ndarray,float,float]] = [
            (np.zeros(self.num_parameters), np.zeros(self.num_parameters*self.num_parameters), 0.0, 1e100)
        ]

        for parameter in parameters:
            if parameter.hard_lower_bound >= parameter.hard_upper_bound:
                raise ValueError(f"{parameter} has touching or crossed hard bounds")

    def _convert_obs(self, param_values: Dict[str,float]) -> np.ndarray:
        """Convert parameters from user coordinates into normalized coordinates"""
        arr = np.zeros(self.num_parameters)
        for i in range(self.num_parameters):
            param = self.parameters[i]
            arr[i] = (param_values[param.name] - param.guess) / param.likely_radius_around_guess
        return arr

    def _unconvert_obs(self, arr: np.ndarray) -> Dict[str,float]:
        """Convert parameters from normalized coordinates into user coordinates"""
        param_values: Dict[str,float] = {}
        for i in range(self.num_parameters):
            param = self.parameters[i]
            param_values[param.name] = arr[i] * param.likely_radius_around_guess + param.guess
        return param_values

    def _unconvert_obs_already_unnormalized(self, arr: np.ndarray) -> Dict[str,float]:
        """Convert parameters from user coordinates into user coordinates but just in dictionary form"""
        param_values: Dict[str,float] = {}
        for i in range(self.num_parameters):
            param = self.parameters[i]
            param_values[param.name] = arr[i]
        return param_values

    def add_win(self, winner: Dict[str,float], loser: Dict[str,float]):
        winner_arr = self._convert_obs(winner)
        loser_arr = self._convert_obs(loser)
        self.observations.append((winner_arr,loser_arr,1.0))
    def add_draw(self, player: Dict[str,float], opponent: Dict[str,float]):
        player_arr = self._convert_obs(player)
        opponent_arr = self._convert_obs(opponent)
        self.observations.append((player_arr,opponent_arr,0.5))
        self.observations.append((opponent_arr,player_arr,0.5))

    def _compute_logwk(self, observation, quadratic_fit: Tuple[np.ndarray,np.ndarray,float,float]):
        linear_coeffs, quadratic_coeffs, logistic_mean, confidence_deviation = quadratic_fit
        numerator = (
            np.dot(observation, linear_coeffs) +
            np.dot(np.outer(observation,observation).flatten(), quadratic_coeffs)
            - logistic_mean
        )
        denominator = (1e-30 + self.locality_stdevs * confidence_deviation)
        return numerator/denominator

    def _compute_min_logwks(self, observation, quadratic_fits: List[Tuple[np.ndarray,np.ndarray,float,float]]):
        return min(self._compute_logwk(observation,quadratic_fit) for quadratic_fit in quadratic_fits)


    def recompute(self):
        num_observations = len(self.observations)

        additional_winner_weight_factors = np.ones(num_observations)
        additional_loser_weight_factors = np.ones(num_observations)
        additional_weight_sum = num_observations * 2
        last_additional_weight_sum = np.inf
        weight_shrink_stop_threshold = num_observations * 2 * self.weight_factor_shrink_stop_threshold

        self.current_quadratic_fits = [
            (np.zeros(self.num_parameters), np.zeros(self.num_parameters*self.num_parameters), 0.0, 1e100)
        ]

        num_iters = 0
        while True:
            if self.debug:
                print("-------------")
                print(f"NUM_ITERS {num_iters} additional_weight_sum {additional_weight_sum} ")
                # print(f"Additional weights {additional_winner_weight_factors} {additional_loser_weight_factors}")
                print(self.current_quadratic_fits[-1])
                try:
                    print("Latest quadratic fit elo covariance: ", np.linalg.inv(-self.current_quadratic_fits[-1][1].reshape(self.num_parameters,self.num_parameters)) * 400.0 * math.log10(math.exp(1)))
                except np.linalg.LinAlgError:
                    pass

            if num_iters >= self.max_iters or additional_weight_sum > last_additional_weight_sum - weight_shrink_stop_threshold:
                break
            num_iters += 1

            additional_weight_factors = (
                (1.0 - self.interp_to_use_max_pair_weight) * np.minimum(additional_winner_weight_factors, additional_loser_weight_factors) +
                self.interp_to_use_max_pair_weight * np.maximum(additional_winner_weight_factors, additional_loser_weight_factors)
            )
            # t0 = time.time()
            linear_coeffs, quadratic_coeffs = quadratic_regress(
                win_loss_weight_datas=self.observations,
                additional_weight_factors=additional_weight_factors,
                initial_linear_coeffs=self.current_quadratic_fits[-1][0],
                initial_quadratic_coeffs=self.current_quadratic_fits[-1][1],
                num_features = self.num_parameters,
                linear_coeff_prior_stdev=self.linear_coeff_prior_stdev,
                quadratic_coeff_prior_stdev=self.quadratic_coeff_prior_stdev,
            )
            # t1 = time.time()
            logistic_mean, confidence_deviation = logistic_mean_and_confidence_deviation(
                win_loss_weight_datas=self.observations,
                additional_winner_weight_factors=additional_winner_weight_factors,
                additional_loser_weight_factors=additional_loser_weight_factors,
                linear_coeffs=linear_coeffs,
                quadratic_coeffs=quadratic_coeffs,
                constant_coeff_prior_stdev=self.constant_coeff_prior_stdev,
            )
            # t2 = time.time()
            # print(t1-t0,t2-t1)

            quadratic_fit = (linear_coeffs, quadratic_coeffs, logistic_mean, confidence_deviation)
            for i,(winner,loser,_) in enumerate(self.observations):
                additional_winner_weight_factors[i] = min(
                    additional_winner_weight_factors[i],
                    math.exp(self._compute_logwk(winner, quadratic_fit))
                )
                additional_loser_weight_factors[i] = min(
                    additional_loser_weight_factors[i],
                    math.exp(self._compute_logwk(loser, quadratic_fit))
                )

            self.current_quadratic_fits.append(quadratic_fit)
            last_additional_weight_sum = additional_weight_sum
            additional_weight_sum = np.sum(additional_winner_weight_factors) + np.sum(additional_loser_weight_factors)

        # for x in np.arange(0.0,4.0,0.1):
        #     print(x, self._compute_logwk(np.array([x]),self.current_quadratic_fits[-1]))

        self.current_optimum = np.zeros(self.num_parameters)
        wsum = 0.0
        for i,(winner,loser,weight) in enumerate(self.observations):
            self.current_optimum += additional_winner_weight_factors[i] * weight * winner
            wsum += additional_winner_weight_factors[i] * weight
            self.current_optimum += additional_loser_weight_factors[i] * weight * loser
            wsum += additional_loser_weight_factors[i] * weight

        self.current_optimum = self.current_optimum / wsum

    def get_current_optimum(self) -> Dict[str,float]:
        return self._unconvert_obs(self.current_optimum)

    def _make_uniform_axis_sampler(self, lbound: float, ubound: float) -> Callable:
        return lambda x: _uniform_strictly_between(lbound,ubound)

    def _make_gaussian_axis_sampler(self, stdev: float) -> Callable:
        return lambda x: random.normalvariate(x, stdev)

    def sample_params_to_evaluate(self) -> Dict[str,float]:
        # Construct axis samplers for gibbs sampling
        # Samplers operate in unnormalized space (i.e. user coordinates)
        axis_samplers = []
        for i in range(self.num_parameters):
            # Look for the quadratic fit with the lowest variance - i.e. highest precision
            largest_precision = 1.0 / (self.parameters[i].likely_radius_around_guess ** 2.0)
            for (linear_coeffs, quadratic_coeffs, logistic_mean, confidence_deviation) in self.current_quadratic_fits:
                precision = quadratic_coeffs.reshape(self.num_parameters,self.num_parameters)[i,i] / (self.parameters[i].likely_radius_around_guess ** 2.0)
                if precision > largest_precision:
                    largest_precision = precision
            stdev = 1.0 / math.sqrt(largest_precision)

            if self.parameters[i].hard_lower_bound is not None and self.parameters[i].hard_upper_bound is not None:
                param_range = self.parameters[i].hard_upper_bound - self.parameters[i].hard_lower_bound
                if stdev * 2 > param_range:
                    axis_sampler = self._make_uniform_axis_sampler(self.parameters[i].hard_lower_bound, self.parameters[i].hard_upper_bound)
                else:
                    axis_sampler = self._make_gaussian_axis_sampler(stdev)
            else:
                axis_sampler = self._make_gaussian_axis_sampler(stdev)
            axis_samplers.append(axis_sampler)

        norm_scale = np.array([parameter.likely_radius_around_guess for parameter in self.parameters])
        norm_offset = np.array([parameter.guess for parameter in self.parameters])

        point_unnormalized = np.copy(self.current_optimum) * norm_scale + norm_offset
        initial_point_unnormalized = np.copy(point_unnormalized)
        current_logweight = self._compute_min_logwks((point_unnormalized-norm_offset)/norm_scale, self.current_quadratic_fits)
        for _ in range(self.num_gibbs_sampling_rounds):
            for i in range(self.num_parameters):
                old_coord = point_unnormalized[i]
                new_coord = axis_samplers[i](old_coord)
                if new_coord <= self.parameters[i].hard_lower_bound or new_coord >= self.parameters[i].hard_upper_bound:
                    continue
                # Metropolis-hasting rejection sampling
                point_unnormalized[i] = new_coord
                new_logweight = self._compute_min_logwks((point_unnormalized-norm_offset)/norm_scale, self.current_quadratic_fits)
                # Regularize sampling a bit based on the user's guess. Standard deviation = 2x the user's guess.
                new_logweight -= 0.125 * (point_unnormalized[i] - initial_point_unnormalized[i]) ** 2.0 / (norm_scale[i] ** 2.0)
                if new_logweight < current_logweight and random.random() >= math.exp(new_logweight - current_logweight):
                    # Reject!
                    point_unnormalized[i] = old_coord
                    continue
                # Accept
                current_logweight = new_logweight
                continue

        ret = self._unconvert_obs_already_unnormalized(point_unnormalized)
        if ret['2'] > 1.0:
            ret['2'] = 1.0

        return self._unconvert_obs_already_unnormalized(point_unnormalized)


if __name__ == "__main__":
    clop = PairwiseCLOP(
        parameters=[
            Parameter(name="a", guess=0.0, likely_radius_around_guess = 1.0, hard_lower_bound = -50.0, hard_upper_bound = 50.0),
            Parameter(name="b", guess=0.0, likely_radius_around_guess = 1.0, hard_lower_bound = -50.0, hard_upper_bound = 50.0),
        ],
        # debug=True,
    )

    def get_true_elo(params):
        return abs(params["a"] - 2.0) * (-50.0) + abs(params["b"] - 1.0) * (-20.0)

    for i in range(100):
        print("======================================================")
        for j in range(10):
            params1 = clop.sample_params_to_evaluate()
            params2 = clop.sample_params_to_evaluate()
            elo1 = get_true_elo(params1)
            elo2 = get_true_elo(params2)
            winner_is_1 = random.random() < 1.0 / (1.0 + 10.0 ** (-(elo1 - elo2)/400.0))
            print(params1, params2, elo1, elo2, winner_is_1)
            if winner_is_1:
                clop.add_win(winner=params1,loser=params2)
            else:
                clop.add_win(winner=params2,loser=params1)

        clop.recompute()
        print("CURRENT OPTIMUM", clop.get_current_optimum())
