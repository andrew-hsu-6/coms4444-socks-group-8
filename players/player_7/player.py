# player 7 - pick how to spend from the budget on day 1, then stick with it.
#
# on the first morning we pick one of four modes for the whole run:
#   cohort - most runs: under $0.60 a roommate a day, when the drawer plus
#            every pack we can afford holds enough wears for the run (7% to
#            spare with 4 sock hands)
#   priced - the drawer will empty whatever we do: give every move a cost in
#            embarrassment points and pick the cheapest, to put that off
#   lavish - plenty of money: toss anything that doesn't match what we wore
#   paced  - some money: toss socks once they've had their share of wears
#
# cohort mode keeps each colour one batch of socks that fade together. with 4
# socks in a hand and at most 3 batches in the drawer, two of them always
# share a batch, so there is always a free pair. to keep it that way:
#   - wear the free pair that pulls towards the batch: the mean of recent
#     sightings in a tight drawer, the nearest group of shades in a roomy one
#   - toss socks a brand new one would match better, so once holes bring in
#     new socks the old batch goes and the colour turns over at once; the
#     tosses may run up to 20% of the budget ahead of an even pace so a
#     turnover happens in one burst instead of trickling
#   - in a roomy drawer with money to spare, toss any sock that helps, and a
#     sock left on its own: it can never be worn free, so it would never
#     rejoin its batch, but six of them come back as one pack of six
#   - with more money, retire each batch at the age the budget can pay for
#     rather than waiting for holes - the same spend as paced mode, but the
#     drawer stays one age instead of every age
#   - always hold back the money for the holes still to come; when holes would
#     outrun it, price moves like priced mode, and near the end of the run just
#     stop tossing so the money lasts

from collections import deque
from dataclasses import dataclass, replace
from itertools import combinations
from math import ceil

from models.player import GameContext, PlayerSnapshot, Selection, TurnContext
from models.player import Player as BasePlayer

PACK_COST = 10.0
SOCK_COST = PACK_COST / 6
FREE_GAP = 6
WHITE_NEW, WHITE_DONE = 255, 127
BLACK_NEW, BLACK_DONE = 0, 64


# ----------------------------------------------------------------------
# tuning
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Knobs:
	# all the numbers we can tweak, with what they do

	# lavish tossing needs at least this much money per roommate per day
	lavish_per_roommate: float = SOCK_COST / 2
	# socks per roommate per day lost to holes once the drawer is worn in
	hole_rate: float = 1 / 34
	# penalty for wearing a worn out sock when we expect to run dry
	dry_penalty: float = 1000.0
	# socks to keep above the bare minimum before we call it running dry
	dry_margin: int = 4
	# never toss a sock with fewer wears than this
	min_toss_wears: int = 2
	# daily fade of old sightings in the drawer memory (ruled modes)
	memory_fade: float = 0.97
	# how many hole replacements' worth of money to hold back
	hole_reserve: float = 1.0
	# a swap has to raise the free match chance by at least this much
	swap_min_gain: float = 0.1
	# swaps allowed per turn when money is tight
	swaps_per_turn: int = 2
	# only swap socks with this many wears (fresh ones blend in by
	# themselves, worn out ones still match the other old socks)
	swap_wears: tuple[int, int] = (3, 30)
	# with tight money keep worn out socks instead of paying to replace them
	keep_worn_when_tight: bool = True
	# stop lavish tossing if the house's pace so far would eat the hole money
	pace_guard: bool = True
	# a drawer is tight when C is below this many times the daily draw
	tight_drawer: float = 1.5
	# ...and a budget is thin below this many dollars per roommate per day
	thin_dollars: float = 0.05
	# paced mode only while each sock must last at most this many wears
	paced_max_wears: float = 25
	# priced mode when, on day 1, each sock must last more than this
	priced_min_wears: float = 13
	# priced mode: points per dollar when the credit bank is empty
	point_per_dollar: float = 3.0
	# priced mode: weight on how much better a fresh sock would fit
	fit_weight: float = 1.0
	# priced mode: daily fade of old sightings
	priced_memory_fade: float = 0.9
	# priced mode: most discard credit we can bank, and use in one turn
	bank_cap: float = 10.0
	priced_tosses_per_turn: int = 2
	# priced mode: packs of money we never touch, in case roommates spend in bursts
	reserve_packs: int = 1
	# cohort mode: when to use it
	# ...only below this many dollars per roommate per day
	cohort_max_dollars: float = 0.6
	# ...and, with 4 sock hands, only when the drawer plus every pack we can
	# buy holds this many times the wears the run needs - near the line cohort
	# mode spends the money the endgame needs, and priced mode stretches it
	# further. 5 sock hands cope with a mixed drawer and don't need the margin
	cohort_supply_margin: float = 1.07

	# cohort mode: which pair to wear
	# the batch is the last few sightings of a colour - more of them when the
	# drawer turns over slowly: this many over the share of the drawer worn
	# each day, for 4 and for 5 sock hands, clamped to the range
	cohort_window_scale: tuple[float, float] = (0.8, 0.5)
	cohort_window_range: tuple[int, int] = (5, 20)
	# with at least this many drawer slots per roommate, pull each sock towards
	# the nearest group of shades we've seen instead of the mean of its colour -
	# in a roomy drawer stragglers last long enough to form groups of their
	# own, in a tight one the groups are just noise
	cohort_nearest_from: float = 8.0

	# cohort mode: which socks to toss
	# sightings of a colour needed before we toss any of it
	cohort_min_seen: int = 3
	# tosses per turn with 4 and with 5 sock hands
	cohort_tosses: tuple[int, int] = (2, 1)
	# a new sock must cut the average charge by more than this...
	cohort_gain: float = 6.0
	# ...or by more than this with money beyond the reserve and the even pace,
	# in a drawer with at least this many slots per roommate (a tight drawer
	# needs that money for its turnovers)
	cohort_rich_gain: float = 0.0
	cohort_rich_from: float = 6.0
	# with 4 sock hands, toss a straggler: a sock the fading memory of recent
	# hands puts fewer than this many others near (and never more than this
	# share of the drawer, so a small drawer's ordinary groups don't count). it
	# can never be worn free, so it never ages back into its batch, but six of
	# them come back as one pack of six matching new socks
	cohort_straggler: float = 5.0
	cohort_straggler_share: float = 0.1
	# ...only with at least this many drawer slots per roommate, and a budget
	# this many times what letting every batch wear out would cost - otherwise
	# the turnovers need the money
	cohort_straggler_from: float = 6.0
	cohort_straggler_ratio: float = 1.35
	# ...and while retiring batches early, never a sock worn this few times: a
	# new batch looks thin until the memory catches up with it
	cohort_straggler_fresh: int = 4
	# with at least this many dollars per roommate per day, don't wait for
	# holes - retire a batch once it's this many times as old as the money can
	# keep replacing it at
	cohort_retire_from: float = 0.2
	cohort_retire: float = 1.3

	# cohort mode: money
	# drawer slots we don't count on, and spare packs held back
	cohort_slack_socks: int = 14
	cohort_spare_packs: int = 1
	# share of a new sock's 68 wears we count on - batches get tossed early
	cohort_life_used: float = 0.9
	# share of the budget our tosses may run ahead of an even pace, so a
	# turnover can happen in one burst instead of trickling
	cohort_burst: float = 0.2
	# with 4 sock hands, over this last share of the run, stop tossing once
	# holes coming this many times faster than the steady rate would run the
	# house dry - a batch wearing out together loses socks in a burst, and
	# there's no time left to recover from a drawer that has started to shrink
	cohort_hole_burst_tail: float = 0.15
	cohort_hole_burst: float = 1.6


TUNED = Knobs()
# with a roomy drawer the plainer rules did better in testing
ROOMY = replace(
	TUNED,
	pace_guard=False,
	keep_worn_when_tight=False,
	swaps_per_turn=1,
	swap_wears=(TUNED.min_toss_wears, 64),
)


# ----------------------------------------------------------------------
# shade arithmetic
# ----------------------------------------------------------------------


def is_white(shade: int) -> bool:
	return shade > BLACK_DONE


def wears(shade: int) -> int:
	# rough wash count - white fades 2 a wash, black goes up 1
	return (WHITE_NEW - shade) // 2 if is_white(shade) else shade


def is_done(shade: int) -> bool:
	# faded all the way, so every wear could put a hole in it
	return shade in (WHITE_DONE, BLACK_DONE)


def charge(a: int, b: int) -> int:
	# what the engine charges for wearing a and b together
	gap = abs(a - b)
	return gap if gap > FREE_GAP else 0


def brand_new(shade: int) -> int:
	return WHITE_NEW if is_white(shade) else BLACK_NEW


def after_wash(shade: int) -> int:
	return max(WHITE_DONE, shade - 2) if is_white(shade) else min(BLACK_DONE, shade + 1)


def colour(shade: int) -> str:
	return 'w' if is_white(shade) else 'b'


def wears_left(shade: float) -> float:
	# wears before it fades out, plus the 4 a worn out sock lasts on average
	# before it gets a hole
	fade = (shade - WHITE_DONE) / 2 if is_white(int(shade)) else BLACK_DONE - shade
	return fade + 4


NEW_SOCK_WEARS = wears_left(WHITE_NEW)


def shade_groups(shades) -> list[list[int]]:
	# split shades wherever sorted neighbours are more than the free gap apart
	groups: list[list[int]] = []
	for shade in sorted(shades):
		if groups and shade - groups[-1][-1] <= FREE_GAP:
			groups[-1].append(shade)
		else:
			groups.append([shade])
	return groups


# ----------------------------------------------------------------------
# what we remember about the drawer
# ----------------------------------------------------------------------


class ShadeMemory:
	# faded counts of the shades we've been handed, per colour

	def __init__(self, fade: float, forget_below: float) -> None:
		self.fade = fade
		self.forget_below = forget_below
		self.counts: dict[str, dict[int, float]] = {'w': {}, 'b': {}}
		self.last_day = 0

	def observe(self, hand: tuple[int, ...], day: int) -> None:
		factor = self.fade ** max(day - self.last_day, 1)
		self.last_day = day
		for counts in self.counts.values():
			for shade in list(counts):
				counts[shade] *= factor
				if counts[shade] < self.forget_below:
					del counts[shade]
		for shade in hand:
			counts = self.counts[colour(shade)]
			counts[shade] = counts.get(shade, 0.0) + 1.0

	def same_colour(self, shade: int) -> dict[int, float]:
		return self.counts[colour(shade)]

	def total(self) -> float:
		return sum(self.counts['w'].values()) + sum(self.counts['b'].values())


class SoloLedger:
	# living alone we can keep track of the whole drawer: shade -> how many.
	# holes are the one thing we can't see, so a worn out sock we wear comes back
	# counted as 0.75 of a sock and the other 0.25 waits for a new pack

	def __init__(self, capacity: int) -> None:
		half = float(capacity // 2)
		self.drawer: dict[int, float] = {WHITE_NEW: half, BLACK_NEW: half}
		self.owed = {'w': 0.0, 'b': 0.0}
		self.spent_seen = 0.0
		self.returning: list[tuple[int, float]] | None = None
		self.lost_today: set[str] = set()

	def morning(self, hand: tuple[int, ...], turn: TurnContext) -> None:
		# last night's socks come back, any new packs go in, today's hand comes out
		for shade, weight in self.returning or ():
			self.drawer[shade] = self.drawer.get(shade, 0.0) + weight
		packs = round((turn.total_spent - self.spent_seen) / PACK_COST)
		self.spent_seen = turn.total_spent
		for _ in range(packs):
			# a pack can only be for a colour we lost yesterday; if both, go by
			# whose count is closer to 6
			lost = [c for c in self.owed if c in self.lost_today] or list(self.owed)
			c = max(lost, key=lambda k: self.owed[k])
			self.owed[c] = max(0.0, self.owed[c] - 6)
			fresh = WHITE_NEW if c == 'w' else BLACK_NEW
			self.drawer[fresh] = self.drawer.get(fresh, 0.0) + 6
		for shade in hand:
			rest = self.drawer.get(shade, 0.0) - 1
			if rest > 0.01:
				self.drawer[shade] = rest
			else:
				self.drawer.pop(shade, None)

	def evening(self, hand: tuple[int, ...], wear: tuple[int, int], toss: list[int]) -> None:
		self.returning = []
		for i, shade in enumerate(hand):
			if i in wear:
				share = 0.75 if is_done(shade) else 1.0
				self.returning.append((after_wash(shade), share))
				self.owed[colour(shade)] += 1.0 - share
			elif i in toss:
				self.owed[colour(shade)] += 1
			else:
				self.returning.append((shade, 1.0))
		self.lost_today = {
			colour(hand[i])
			for i in range(len(hand))
			if i in toss or (i in wear and is_done(hand[i]))
		}

	def same_colour(self, shade: int) -> dict[int, float]:
		white = is_white(shade)
		return {s: w for s, w in self.drawer.items() if is_white(s) == white}

	def total(self) -> float:
		return sum(self.drawer.values())


# ----------------------------------------------------------------------
# money
# ----------------------------------------------------------------------


def hole_money(k: Knobs, roommates: int, days: int) -> float:
	# money to keep back for replacing holes over this many days
	return k.hole_reserve * k.hole_rate * roommates * days * SOCK_COST


def socks_per_day(k: Knobs, money: float, roommates: int, days: int) -> float:
	# new socks a day the house can afford after setting aside the hole money
	return (money - hole_money(k, roommates, days)) / days / SOCK_COST


@dataclass
class Wallet:
	# where the house's money stands this morning

	have_money: bool = True
	lavish: bool = True
	running_dry: bool = False
	# wears each sock should last when spending evenly (None: not pacing)
	share: float | None = None


# ----------------------------------------------------------------------
# the player
# ----------------------------------------------------------------------


class Player7(BasePlayer):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.regime: str | None = None
		self.knobs = TUNED
		self.memory = ShadeMemory(TUNED.memory_fade, 0.01)
		self.solo = SoloLedger(self.capacity) if self.roommates == 1 else None
		# ruled modes
		self.broke_since: int | None = None
		self.swap_credit = 0.0
		# priced mode
		self.bank = 0.0
		self.my_spend = 0.0
		self.target_wears = 64.0
		self.trust_target = 0.0
		# cohort mode
		self.recent: dict[str, deque[int]] = {}

	def select_socks(self, offered: tuple[int, ...], turn: TurnContext) -> Selection:
		if self.regime is None:
			self.settle_regime(turn)
		if self.regime == 'cohort':
			return self.cohort_turn(offered, turn)
		if self.regime == 'priced':
			return self.priced_turn(offered, turn)
		return self.ruled_turn(offered, turn)

	def settle_regime(self, turn: TurnContext) -> None:
		# decided once, from the whole budget over the whole run
		k = TUNED
		budget = turn.total_spent + turn.budget_remaining
		if self.use_cohort(budget):
			self.regime = 'cohort'
			scale = k.cohort_window_scale[self.selection_unit >= 5]
			lo, hi = k.cohort_window_range
			size = max(lo, min(hi, round(scale * self.capacity / (2 * self.roommates))))
			self.recent = {'w': deque(maxlen=size), 'b': deque(maxlen=size)}
			self.memory = ShadeMemory(k.priced_memory_fade, 1e-3)
			return
		if budget != float('inf'):
			rate = socks_per_day(k, budget, self.roommates, self.days)
			if rate <= 0 or 2 * self.roommates / rate > k.priced_min_wears:
				self.regime = 'priced'
				self.memory = ShadeMemory(k.priced_memory_fade, 1e-3)
				return
		self.regime = 'ruled'
		tight = self.capacity / (self.selection_unit * self.roommates) < k.tight_drawer
		thin = budget / (self.roommates * self.days) < k.thin_dollars
		self.knobs = TUNED if tight or thin else ROOMY
		self.memory = ShadeMemory(k.memory_fade, 0.01)

	# ------------------------------------------------------------------
	# lavish / paced: fixed rules
	# ------------------------------------------------------------------

	def ruled_turn(self, hand: tuple[int, ...], turn: TurnContext) -> Selection:
		self.memory.observe(hand, turn.day)
		if self.solo:
			self.solo.morning(hand, turn)

		wallet, k = self.check_wallet(turn)
		wear = self.closest_pair(hand, avoid_done=wallet.running_dry, k=k)
		rest = [i for i in range(len(hand)) if i not in wear]

		share = wallet.share
		spending = wallet.have_money and not wallet.running_dry and not wallet.lavish
		if share is not None and spending:
			toss = [i for i in rest if wears(hand[i]) >= max(share, k.min_toss_wears)]
			return self.finish(hand, wear, toss)

		toss = self.rule_tosses(hand, wear, rest, wallet, k)
		if wallet.have_money and not wallet.lavish and not wallet.running_dry:
			toss += self.swaps(hand, rest, toss, k)
		return self.finish(hand, wear, toss)

	def check_wallet(self, turn: TurnContext) -> tuple[Wallet, Knobs]:
		k, n = self.knobs, self.roommates
		days_left = max(self.days - turn.day + 1, 1)
		left = turn.budget_remaining
		if left == float('inf'):
			return Wallet(), k

		wallet = Wallet()
		rate = socks_per_day(TUNED, left, n, days_left)
		if rate > 0 and 2 * n / rate <= TUNED.paced_max_wears:
			wallet.share = 2 * n / rate
			k = TUNED

		wallet.have_money = left >= PACK_COST
		if not wallet.have_money and self.broke_since is None:
			self.broke_since = turn.day
		wallet.lavish = wallet.have_money and left / days_left >= k.lavish_per_roommate * n
		if wallet.lavish and k.pace_guard and wallet.share is None and turn.day > 20:
			# the house's spending pace so far, kept up to the end, must still
			# leave the money we need for holes
			pace = turn.total_spent / turn.day
			wallet.lavish = pace * days_left <= left - hole_money(k, n, days_left)
		wallet.running_dry = self.running_dry(turn, left if wallet.have_money else 0.0, days_left)

		# whatever isn't needed for holes is spare: spread it over the days
		# left and split it between roommates so copies of us don't each
		# spend all of it
		spare = left - hole_money(k, n, days_left)
		if spare > 0 and not wallet.running_dry:
			self.swap_credit = min(self.swap_credit + spare / days_left / n / SOCK_COST, 3.0)
		return wallet, k

	def running_dry(
		self, turn: TurnContext, left: float, days_left: int, burst: float = 1.0
	) -> bool:
		# will holes from here on eat the spare socks plus what money can replace?
		k, n = TUNED, self.roommates
		lost = 0.0
		if self.broke_since is not None:
			lost = k.hole_rate * n * (turn.day - self.broke_since)
		spare = self.capacity - lost - self.selection_unit * n - k.dry_margin
		return burst * k.hole_rate * n * days_left > spare + left / SOCK_COST

	@staticmethod
	def closest_pair(hand: tuple[int, ...], avoid_done: bool, k: Knobs) -> tuple[int, int]:
		# cheapest pair, fresher if tied; when running dry every hole is a
		# sock we never get back, so stay off worn out socks
		penalty = k.dry_penalty if avoid_done else 0.0

		def score(pair: tuple[int, int]) -> tuple[float, int]:
			a, b = hand[pair[0]], hand[pair[1]]
			return (charge(a, b) + penalty * (is_done(a) + is_done(b)), wears(a) + wears(b))

		return min(combinations(range(len(hand)), 2), key=score)

	@staticmethod
	def rule_tosses(
		hand: tuple[int, ...], wear: tuple[int, int], rest: list[int], wallet: Wallet, k: Knobs
	) -> list[int]:
		worn_shade = (hand[wear[0]] + hand[wear[1]]) / 2
		toss = []
		for i in rest:
			shade = hand[i]
			if is_done(shade):
				# replace worn out socks while we can, unless every sock counts
				# or money is tight (they still match the other old ones)
				keep = wallet.running_dry or (not wallet.lavish and k.keep_worn_when_tight)
				if wallet.have_money and not keep:
					toss.append(i)
			elif wallet.lavish and shade != worn_shade and wears(shade) >= k.min_toss_wears:
				# plenty of money: drop the odd ones out
				toss.append(i)
		return toss

	def swaps(self, hand: tuple[int, ...], rest: list[int], toss: list[int], k: Knobs) -> list[int]:
		# tight money: spend banked credit on the mid-aged socks that fit in
		# worst, worst first
		lo, hi = k.swap_wears
		picks = [i for i in rest if i not in toss and lo <= wears(hand[i]) <= hi]
		picks.sort(key=lambda i: -self.swap_gain(hand[i]))
		chosen = []
		for i in picks[: k.swaps_per_turn]:
			if self.swap_credit < 1.0 or self.swap_gain(hand[i]) < k.swap_min_gain:
				break
			chosen.append(i)
			self.swap_credit -= 1.0
		return chosen

	def swap_gain(self, shade: int) -> float:
		# how much more often a brand new sock would find a free match
		return self.match_chance(brand_new(shade)) - self.match_chance(shade)

	def match_chance(self, shade: int) -> float:
		# chance another sock in the hand is within the free gap of this one
		source = self.solo or self.memory
		total = source.total()
		if total == 0:
			return 1.0
		near = sum(w for s, w in source.same_colour(shade).items() if abs(s - shade) <= FREE_GAP)
		return 1.0 - (1.0 - near / total) ** (self.selection_unit - 1)

	def finish(self, hand: tuple[int, ...], wear: tuple[int, int], toss: list[int]) -> Selection:
		if self.solo:
			self.solo.evening(hand, wear, toss)
		return Selection(wear=wear, discard=tuple(toss))

	# ------------------------------------------------------------------
	# priced: cheapest move wins
	# ------------------------------------------------------------------

	def priced_turn(
		self, hand: tuple[int, ...], turn: TurnContext, seen: bool = False
	) -> Selection:
		k, n = TUNED, self.roommates
		if not seen:
			self.memory.observe(hand, turn.day)
		days_left = max(self.days - turn.day + 1, 1)
		left = turn.budget_remaining
		broke = left < PACK_COST

		tosses_allowed = self.fill_bank(turn, left, days_left) if not broke else 0

		# the drawer our budget is steering towards: ages spread evenly up to
		# the wear count we can afford, trusted more the faster we replace
		rate = max(0.0, socks_per_day(k, left, n, days_left)) + k.hole_rate * n
		self.target_wears = 2 * n / rate
		self.trust_target = min(1.0, rate * min(30, days_left) / self.capacity)

		# money is cheap while the bank is full, priceless once it's gone
		per_dollar = k.point_per_dollar * max(0.0, 1.0 - self.bank / k.bank_cap)
		if broke:
			per_dollar = 1000.0
		new_sock = per_dollar * SOCK_COST

		# wearing a worn out sock is a 25% chance of paying for a new one;
		# tossing pays for one but trades this sock's fit for a fresh one's
		wear_price = [0.25 * new_sock * is_done(s) for s in hand]
		toss_price = [
			new_sock + k.fit_weight * (self.misfit(brand_new(s)) - self.misfit(s)) for s in hand
		]

		def move(pair: tuple[int, int]) -> tuple[tuple[float, int, int], tuple[int, ...]]:
			i, j = pair
			worth_it = sorted(
				(toss_price[x], x) for x in range(len(hand)) if x not in pair and toss_price[x] < 0
			)[:tosses_allowed]
			price = charge(hand[i], hand[j]) + wear_price[i] + wear_price[j]
			price += sum(p for p, _ in worth_it)
			return (price, len(worth_it), wears(hand[i]) + wears(hand[j])), tuple(
				sorted(x for _, x in worth_it)
			)

		moves = [(pair, *move(pair)) for pair in combinations(range(len(hand)), 2)]
		wear, _, toss = min(moves, key=lambda m: m[1])
		self.bank -= len(toss)
		self.my_spend += len(toss) * SOCK_COST
		return Selection(wear=wear, discard=toss)

	def fill_bank(self, turn: TurnContext, left: float, days_left: int) -> int:
		# put our share of today's spare money in the bank and say how many tosses
		# we get today. what the house spent minus what we spent is how fast the
		# roommates spend - assume they keep that up, keep a reserve, rest is ours
		k = TUNED
		if days_left <= 15:
			return 0
		others = max(0.0, turn.total_spent - self.my_spend) / max(turn.day, 1)
		reserve = max(k.reserve_packs * PACK_COST, 0.03 * (turn.total_spent + left))
		slack = left - reserve - others * days_left
		if slack <= 0:
			self.bank = 0.0
			return 0
		self.bank = min(self.bank + slack / SOCK_COST / days_left, k.bank_cap)
		return min(k.priced_tosses_per_turn, int(self.bank))

	def misfit(self, shade: int) -> float:
		# how badly this shade would match a random sock of the same colour -
		# partly the drawer we see now, partly the one our spending is heading for
		seen = self.memory.same_colour(shade)
		weight = sum(seen.values())
		now = sum(w * charge(shade, s) for s, w in seen.items()) / weight if weight else 0.0
		top = min(64, max(1, int(self.target_wears)))
		new, step = (WHITE_NEW, -2) if is_white(shade) else (BLACK_NEW, 1)
		later = sum(charge(shade, new + step * a) for a in range(top + 1)) / (top + 1)
		return (1 - self.trust_target) * now + self.trust_target * later

	# ------------------------------------------------------------------
	# cohort
	# ------------------------------------------------------------------

	def use_cohort(self, budget: float) -> bool:
		# cohort mode spends close to the least it can, which is right when
		# money is tight but wastes it when lavish tossing is affordable
		if budget / (self.roommates * self.days) >= TUNED.cohort_max_dollars:
			return False
		# if the drawer plus every pack we can buy can't cover the run's wears,
		# the drawer will empty whatever we do, and priced mode is better at
		# putting that off
		supply = (self.capacity + budget // PACK_COST * 6) * NEW_SOCK_WEARS
		margin = TUNED.cohort_supply_margin if self.selection_unit == 4 else 1.0
		return supply >= margin * 2 * self.roommates * self.days

	def cohort_turn(self, hand: tuple[int, ...], turn: TurnContext) -> Selection:
		for shade in hand:
			self.recent[colour(shade)].append(shade)
		self.memory.observe(hand, turn.day)
		left = turn.budget_remaining
		if left < PACK_COST and self.broke_since is None:
			self.broke_since = turn.day
		days_left = max(self.days - turn.day + 1, 1)
		money = left if left >= PACK_COST else 0.0
		if self.running_dry(turn, money, days_left):
			# holes will shrink the drawer faster than money can refill it and a
			# sockless day costs more than any mismatch, so price every move
			# the way priced mode does until the money catches up again
			return self.priced_turn(hand, turn, seen=True)
		# near the end a batch wearing out loses socks in a burst, so judge dry
		# with a faster hole rate there
		tail = self.selection_unit == 4 and days_left <= TUNED.cohort_hole_burst_tail * self.days
		if tail and self.running_dry(turn, money, days_left, TUNED.cohort_hole_burst):
			# keep the money for the holes and keep every sock, but pair as usual
			return Selection(wear=self.cohort_pair(hand), discard=())
		wear = self.cohort_pair(hand)
		rest = [i for i in range(len(hand)) if i not in wear]
		return Selection(wear=wear, discard=self.cohort_tosses(hand, rest, turn))

	def cohort_pair(self, hand: tuple[int, ...]) -> tuple[int, int]:
		# of the free pairs, the one whose wash best pulls its socks towards
		# their batch; with no free pair, the smallest mismatch
		pairs = list(combinations(range(len(hand)), 2))
		free = [p for p in pairs if charge(hand[p[0]], hand[p[1]]) == 0]
		if not free:
			return min(pairs, key=lambda p: abs(hand[p[0]] - hand[p[1]]))

		centres = {c: self.batch_centre(c) for c in ('w', 'b')}
		nearest = self.capacity / self.roommates >= TUNED.cohort_nearest_from
		groups = {c: shade_groups(self.recent[c]) for c in ('w', 'b')}

		def target(shade: int) -> float:
			if nearest and groups[colour(shade)]:
				# the group of recent sightings this sock sits closest to
				near = min(groups[colour(shade)], key=lambda g: min(abs(x - shade) for x in g))
				return sum(near) / len(near)
			return centres[colour(shade)]

		def drift(pair: tuple[int, int]) -> float:
			# how much washing these two moves them away from their batch -
			# negative means it pulls stragglers back in
			moved = 0.0
			for i in pair:
				centre = target(hand[i])
				moved += (after_wash(hand[i]) - centre) ** 2 - (hand[i] - centre) ** 2
			return moved

		return min(free, key=lambda p: (drift(p), abs(hand[p[0]] - hand[p[1]]), p))

	def batch_centre(self, c: str) -> float:
		seen = self.recent[c]
		if not seen:
			return WHITE_NEW if c == 'w' else BLACK_NEW
		return sum(seen) / len(seen)

	def cohort_tosses(self, hand: tuple[int, ...], rest: list[int], turn: TurnContext) -> tuple:
		# which of the socks we aren't wearing to throw out, best first
		if not rest or not self.cohort_can_spend(turn):
			return ()
		k = TUNED
		bar = self.toss_bar(turn)
		retire_at = self.retire_age(turn)
		lonely = self.straggler_cut(turn)
		# while retiring batches early, a new batch looks thin until the memory
		# catches up with it - leave those socks alone
		fresh = -1 if retire_at is None else k.cohort_straggler_fresh
		picks = []
		for i in rest:
			shade = hand[i]
			seen = self.recent[colour(shade)]
			if len(seen) < k.cohort_min_seen:
				continue
			new = brand_new(shade)
			gain = sum(charge(shade, s) - charge(new, s) for s in seen) / len(seen)
			if gain > bar:
				picks.append((-gain, i))
			elif lonely and wears(shade) > fresh and self.company(shade) < lonely:
				picks.append((0.0, i))
			elif retire_at is not None and wears(shade) >= retire_at:
				# the batch is as old as the money says it may get: start the
				# turnover, and the gain rule clears the rest once new socks show
				picks.append((0.0, i))
		toss, lost = [], 0.0
		for _, i in sorted(picks):
			# every sock we toss is wears the house has to buy back
			if turn.budget_remaining < self.cohort_reserve(turn, lost + wears_left(hand[i])):
				continue
			toss.append(i)
			lost += wears_left(hand[i])
			if len(toss) >= k.cohort_tosses[self.selection_unit >= 5]:
				break
		return tuple(sorted(toss))

	def toss_bar(self, turn: TurnContext) -> float:
		# how much a new sock must cut the average charge before we toss for it:
		# anything at all in a roomy drawer with money beyond the reserve and the
		# even pace, otherwise a clear improvement
		k = TUNED
		budget = turn.total_spent + turn.budget_remaining
		surplus = turn.budget_remaining - self.cohort_reserve(turn, 0.0)
		if budget != float('inf'):
			surplus -= budget * (self.days - turn.day) / self.days
		rich = surplus > PACK_COST * self.roommates
		roomy = self.capacity / self.roommates >= k.cohort_rich_from
		return k.cohort_rich_gain if rich and roomy else k.cohort_gain

	def retire_age(self, turn: TurnContext) -> float | None:
		# with enough money, the wear count at which to start retiring a batch:
		# a whole batch replaced every A wears costs about 2n/A socks a day - the
		# same money as paced mode, but the drawer stays one age
		k = TUNED
		budget = turn.total_spent + turn.budget_remaining
		if budget / (self.roommates * self.days) < k.cohort_retire_from:
			return None
		days_left = max(self.days - turn.day + 1, 1)
		per_day = turn.budget_remaining / days_left / SOCK_COST
		if per_day <= 0:
			return None
		return max(k.min_toss_wears, k.cohort_retire * 2 * self.roommates / per_day)

	def straggler_cut(self, turn: TurnContext) -> float:
		# a sock with fewer than this many others near its shade is a straggler
		# worth tossing - 0 when we don't toss stragglers in this run
		k = TUNED
		if self.selection_unit != 4 or self.capacity / self.roommates < k.cohort_straggler_from:
			return 0.0
		budget = turn.total_spent + turn.budget_remaining
		if budget != float('inf'):
			worn_out = 2 * self.roommates * self.days / NEW_SOCK_WEARS - self.capacity / 2
			if budget < k.cohort_straggler_ratio * worn_out * SOCK_COST:
				return 0.0
		return min(k.cohort_straggler, k.cohort_straggler_share * self.capacity)

	def company(self, shade: int) -> float:
		# roughly how many socks of this colour sit within the free gap of it
		seen = self.memory.same_colour(shade)
		total = sum(seen.values())
		if not total:
			return float(self.capacity)
		near = sum(w for s, w in seen.items() if abs(s - shade) <= FREE_GAP)
		return near / total * self.capacity / 2

	def cohort_can_spend(self, turn: TurnContext) -> bool:
		left = turn.budget_remaining
		if left < PACK_COST or left < self.cohort_reserve(turn, 0.0):
			return False
		budget = turn.total_spent + left
		if budget == float('inf'):
			return True
		# don't get ahead of an even pace through the budget
		return left / budget > (self.days - turn.day) / self.days - TUNED.cohort_burst

	def cohort_reserve(self, turn: TurnContext, lost: float) -> float:
		# money for the packs the house will need to finish the run, if the
		# drawer we see now is worn down to holes, plus some spare
		k = TUNED
		per_colour = max(0.0, (self.capacity - max(10, k.cohort_slack_socks)) / 2)
		supply = 0.0
		for seen in self.recent.values():
			life = sum(wears_left(s) for s in seen) / len(seen) if seen else NEW_SOCK_WEARS
			supply += per_colour * life
		supply = max(0.0, supply - lost)
		need = 2 * self.roommates * (self.days - turn.day + 1)
		packs = ceil(max(0.0, need - supply) / (k.cohort_life_used * NEW_SOCK_WEARS * 6))
		return PACK_COST * (packs + k.cohort_spare_packs)
