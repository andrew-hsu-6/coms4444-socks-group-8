import math
from collections import deque
from itertools import combinations
from statistics import median

from core.engine import PACK_COST
from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

THRESHOLD = 6
AGE_CUT_LOW = 3
AGE_CUT_MID = 7
AGE_CUT_HIGH = 15
USABLE_FRAC = 0.85


class Player10(BasePlayer):
	"""Group 10 sock-selection strategy."""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

		self.days_seen = 0

		# --------------------------------------------------
		# ADAPTIVE HISTORY SIZE
		# --------------------------------------------------
		#
		# A sock is worn approximately:
		#
		#     2 * roommates / capacity
		#
		# times per day.
		#
		# We want our observations to cover roughly C/n days,
		# which corresponds to approximately two expected wears
		# per sock.
		#
		# Since we observe selection_unit socks each day:
		#
		#     history_size ~= (C / n) * selection_unit
		#
		self.history_size = 5

		self.recent_black = deque(maxlen=self.history_size)
		self.recent_white = deque(maxlen=self.history_size)

		# Don't trust the estimated distribution until we have
		# observed a reasonable amount of data for that colour.
		self.min_observations = 3

	def aging(self, shade: int) -> float:
		"""Estimate how many times a sock has been worn."""

		if shade >= 127:  # white sock
			return (255 - shade) / 2
		return shade

	def is_white(self, shade: int) -> bool:
		"""Determine the original colour of a sock."""
		return shade > 64

	def mismatch(self, shade1: int, shade2: int) -> int:
		"""
		Return the embarrassment-style mismatch between two socks.

		A difference of 6 or less costs zero embarrassment.
		"""
		difference = abs(shade1 - shade2)

		if difference <= THRESHOLD:
			return 0

		return difference

	def average_mismatch(self, shade: int, observations) -> float:
		"""
		Calculate the average mismatch between a sock and a collection
		of recently observed same-colour socks.
		"""
		return sum(self.mismatch(shade, observed) for observed in observations) / len(observations)

	def distribution_stats(self, observations) -> tuple[float, float]:
		"""
		Estimate the center and spread of a same-colour distribution.

		The center is the median shade.

		The spread is the median absolute deviation (MAD), which is less
		sensitive to extreme socks than standard deviation.
		"""
		center = median(observations)

		deviations = [abs(shade - center) for shade in observations]

		spread = median(deviations)

		return center, spread

	def outlier_threshold(self, observations) -> float:
		"""
		Determine how far a sock must be from the observed distribution
		before we consider it unusually far away.

		Narrow distribution:
		threshold approaches the game's natural threshold of 6.

		Wide distribution:
		threshold increases automatically.
		"""
		_, spread = self.distribution_stats(observations)

		return max(
			THRESHOLD,
			3 * spread,
		)

	def remaining_wears(self, shade: int) -> float:
		"""
		Estimate expected future wears before a sock develops a hole.

		White: loses 2 shade units per wear until 127.

		Black: gains 1 shade unit per wear until 64.

		At the terminal shade, each wear has a 25% hole probability,
		so expected additional wears are approximately 4.
		"""
		if self.is_white(shade):
			wears_until_terminal = max(
				(shade - 127) / 2,
				0,
			)
		else:
			wears_until_terminal = max(
				64 - shade,
				0,
			)

		return wears_until_terminal + 4

	def estimate_budget_reserve(self, days_remaining: int) -> float:
		"""
		Estimate money that should be reserved for future hole-driven
		replacements, plus two extra safety packs.

		This is only an approximation because we cannot see the complete
		drawer or the engine's discard counters.
		"""

		# Following Group 2's approximation, assume roughly C - 4 socks
		# are available and split evenly between the two colours.
		socks_per_colour = max(
			(self.capacity - 10) / 2,
			0,
		)

		# Estimate remaining useful wears for white socks.
		if self.recent_white:
			avg_white_wears = sum(self.remaining_wears(shade) for shade in self.recent_white) / len(
				self.recent_white
			)
		else:
			# A pristine sock has approximately:
			# 64 wears to terminal shade + 4 expected terminal wears.
			avg_white_wears = 68

		# Estimate remaining useful wears for black socks.
		if self.recent_black:
			avg_black_wears = sum(self.remaining_wears(shade) for shade in self.recent_black) / len(
				self.recent_black
			)
		else:
			avg_black_wears = 68

		estimated_available_wears = (
			socks_per_colour * avg_white_wears + socks_per_colour * avg_black_wears
		)

		# The household wears two socks per roommate per day.
		future_wears_needed = 2 * self.roommates * days_remaining

		wears_missing = max(
			future_wears_needed - estimated_available_wears,
			0,
		)

		# One pack gives six new socks.
		# Each new sock contributes approximately 68 expected wears.
		wears_per_pack = 6 * 68

		future_packs_needed = math.ceil(wears_missing / wears_per_pack)

		# Reserve money for predicted future packs plus two safety packs.
		return future_packs_needed * PACK_COST + 2 * PACK_COST

	def select_socks(
		self,
		offered: tuple[int, ...],
		turn: TurnContext,
	) -> Selection:
		"""Choose two socks to wear and decide whether to discard a leftover."""

		self.days_seen += 1

		# ==================================================
		# STEP 1: CHOOSE WHICH SOCKS TO WEAR
		# ==================================================

		all_pairs = list(combinations(range(len(offered)), 2))

		def embarrassment(pair) -> int:
			difference = abs(offered[pair[0]] - offered[pair[1]])

			# All differences <= 6 have zero embarrassment.
			return 0 if difference <= THRESHOLD else difference

		# Find the lowest possible embarrassment this turn.
		min_embarrassment = min(embarrassment(pair) for pair in all_pairs)

		# Keep every pair tied for that minimum embarrassment.
		best_pairs = [pair for pair in all_pairs if embarrassment(pair) == min_embarrassment]

		# Among equally good embarrassment choices:
		#
		# 1. Prefer the NEWEST pair (smallest total aging).
		# 2. If still tied, prefer the pair whose average
		#    shade is closest to 0.
		def spread_change(pair):
			change = 0.0
			for index in pair:
				shade = offered[index]
				history = self.recent_white if self.is_white(shade) else self.recent_black
				observations = list(history) + [
					x for x in offered if self.is_white(x) == self.is_white(shade)
				]
				center = sum(observations) / len(observations)
				aged = max(127, shade - 2) if self.is_white(shade) else min(64, shade + 1)
				change += (aged - center) ** 2 - (shade - center) ** 2
			return change

		i, j = min(
			best_pairs,
			key=lambda pair: (spread_change(pair), abs(offered[pair[0]] - offered[pair[1]]), pair),
		)

		# ==================================================
		# STEP 2: IDENTIFY LEFTOVER SOCKS
		# ==================================================

		leftovers = [k for k in range(len(offered)) if k not in (i, j)]

		discard: list[int] = []

		days_remaining = max(
			self.days - turn.day + 1,
			1,
		)

		# ==================================================
		# STEP 3: BUDGET PROTECTION
		# ==================================================

		# Group 2 disables voluntary discards when five socks
		# are being selected.
		voluntary_discard_allowed = self.selection_unit < 5

		if turn.budget_remaining is None or turn.budget_remaining == float('inf'):
			# Unlimited budget.
			budget_safe = True

		else:
			# No voluntary spending if another pack cannot
			# even be afforded.
			if turn.budget_remaining < PACK_COST:
				budget_safe = False

			else:
				# Estimate how much money should be protected
				# for future unavoidable replacements.
				reserve = self.estimate_budget_reserve(days_remaining)

				# Original household budget can be reconstructed as:
				#
				# spent so far + budget remaining
				total_budget = turn.total_spent + turn.budget_remaining

				if total_budget > 0:
					budget_fraction_remaining = turn.budget_remaining / total_budget
				else:
					budget_fraction_remaining = 0

				time_fraction_remaining = days_remaining / self.days

				# Only voluntarily discard when our remaining
				# budget fraction is more than five percentage
				# points ahead of the remaining time fraction.
				ahead_of_schedule = budget_fraction_remaining > time_fraction_remaining + 0.05

				# Keep enough money for estimated future hole
				# replacements plus another pack before choosing
				# to spend voluntarily.
				has_reserve = turn.budget_remaining >= reserve + PACK_COST

				budget_safe = ahead_of_schedule and has_reserve

		# ==================================================
		# STEP 4: EVALUATE LEFTOVER SOCKS
		# ==================================================

		if voluntary_discard_allowed and budget_safe and days_remaining > 1:
			candidates = []

			for k in leftovers:
				shade = offered[k]

				# Compare the sock only against observations
				# of its own original colour.
				if self.is_white(shade):
					recent = self.recent_white
					pristine = 255
				else:
					recent = self.recent_black
					pristine = 0

				# Do not trust the estimated distribution
				# until enough same-colour socks have been seen.
				if len(recent) < self.min_observations:
					continue

				# ------------------------------------------
				# Is this sock unusually far from the
				# observed same-colour population?
				# ------------------------------------------

				# Adaptive threshold:
				#
				# At least 6, but larger when the observed
				# distribution itself is broad.

				# If the sock fits comfortably inside the
				# observed population, keep it.

				# ------------------------------------------
				# Would replacing it actually improve
				# future matching?
				# ------------------------------------------

				current_mismatch = self.average_mismatch(
					shade,
					recent,
				)

				replacement_mismatch = self.average_mismatch(
					pristine,
					recent,
				)

				improvement = current_mismatch - replacement_mismatch

				# The sock must not merely be unusual.
				#
				# Replacing it with a pristine sock must
				# actually improve expected matching by
				# more than the game's threshold.
				if improvement > THRESHOLD:
					candidates.append((improvement, k))

			# Group 2 discards at most ONE voluntary sock
			# per turn.
			if candidates:
				_, worst_index = max(candidates)

				discard.append(worst_index)

		# ==================================================
		# STEP 5: UPDATE OUR OBSERVATION HISTORY
		# ==================================================
		#
		# Do this AFTER today's decision.
		#
		# Otherwise, a leftover sock would be included in the
		# distribution used to judge itself.

		for shade in offered:
			if self.is_white(shade):
				self.recent_white.append(shade)
			else:
				self.recent_black.append(shade)

		# ==================================================
		# STEP 6: RETURN DECISION
		# ==================================================

		return Selection(
			wear=(i, j),
			discard=tuple(discard),
		)
