import solvers.«7.py.lean.d».Source

/-! Bounds for the retry schedule used by the Python source. `FirstSuccess`
records both the successful attempt and every preceding failed attempt; the
source correctness proof supplies the same `nthPrimeWithLimit` decisions. -/
namespace solvers.«7».complexity
open ProjectEulerSolutions.P7
open scoped BigOperators

/-- Limit used after exactly `retries` executions of `limit *= 2`. -/
def attemptLimit (initial retries : Nat) : Nat := initial * 2 ^ retries

theorem attemptLimit_zero (initial : Nat) : attemptLimit initial 0 = initial := by
  simp [attemptLimit]

theorem attemptLimit_succ (initial retries : Nat) :
    attemptLimit initial (retries + 1) = attemptLimit initial retries * 2 := by
  simp [attemptLimit, pow_succ, Nat.mul_assoc]

def FirstSuccess (k initial retries : Nat) : Prop :=
  nthPrimeWithLimit k (attemptLimit initial retries) = some (Nat.nth Nat.Prime k) ∧
  ∀ j < retries, nthPrimeWithLimit k (attemptLimit initial j) = none

/-- A prime bound supplies a successful attempt without an assumption about
how accurate the integer starting estimate is. -/
theorem firstSuccess_exists (k initial fuel : Nat)
    (enough : Nat.nth Nat.Prime k ≤ attemptLimit initial fuel) :
    ∃ retries ≤ fuel, FirstSuccess k initial retries := by
  let success := fun r => nthPrimeWithLimit k (attemptLimit initial r) = some (Nat.nth Nat.Prime k)
  have existsSuccess : ∃ r, success r := ⟨fuel, nthPrimeWithLimit_eq_nth k _ enough⟩
  refine ⟨Nat.find existsSuccess, Nat.find_min' existsSuccess (nthPrimeWithLimit_eq_nth k _ enough),
    Nat.find_spec existsSuccess, ?_⟩
  intro j earlier
  cases result : nthPrimeWithLimit k (attemptLimit initial j) with
  | none => rfl
  | some prime =>
      have correct := nthPrimeWithLimit_sound result
      subst prime
      exact False.elim (Nat.find_min existsSuccess earlier result)

/-- At most `n` retries and `n + 1` attempts for ordinal `n ≥ 1`.
Bertrand's theorem supplies the conservative numeric bound `p_n ≤ 2^n`. -/
theorem source_retry_count (k : Nat) :
    ∃ retries ≤ k + 1, FirstSuccess k (initialLimit (k + 1)) retries := by
  apply firstSuccess_exists
  calc
    Nat.nth Nat.Prime k ≤ 2 ^ (k + 1) := nth_prime_le_two_pow_succ k
    _ ≤ attemptLimit (initialLimit (k + 1)) (k + 1) := by
      simpa [attemptLimit] using Nat.mul_le_mul_right (2 ^ (k + 1)) (one_le_initialLimit (k + 1))

/-- A failed attempt has limit below the requested prime. -/
theorem failed_limit_lt (k limit : Nat) (failed : nthPrimeWithLimit k limit = none) :
    limit < Nat.nth Nat.Prime k := by
  by_contra bad
  have succeeds := nthPrimeWithLimit_eq_nth k limit (by omega)
  rw [failed] at succeeds
  contradiction

/-- Doubling can overshoot the target by less than a factor of two. The
starting estimate is retained explicitly, including an oversized estimate. -/
theorem firstSuccess_limit (run : FirstSuccess k initial retries) :
    attemptLimit initial retries ≤ max initial (2 * Nat.nth Nat.Prime k) := by
  cases retries with
  | zero => simp [attemptLimit]
  | succ r =>
      have previous := failed_limit_lt k _ (run.2 r (by omega))
      rw [attemptLimit_succ]
      have twice : attemptLimit initial r * 2 ≤ 2 * Nat.nth Nat.Prime k := by omega
      exact twice.trans (Nat.le_max_right _ _)

/-- Exact geometric accounting for all attempts, including the successful one. -/
theorem attemptLimit_sum (initial retries : Nat) :
    (∑ j ∈ Finset.range (retries + 1), attemptLimit initial j) + initial =
      2 * attemptLimit initial retries := by
  induction retries with
  | zero => simp [attemptLimit]; omega
  | succ r ih =>
      rw [Finset.sum_range_succ, attemptLimit_succ]
      omega

/-- Retries do not multiply linear per-attempt work by the retry count. -/
theorem firstSuccess_total_limits (run : FirstSuccess k initial retries) :
    (∑ j ∈ Finset.range (retries + 1), attemptLimit initial j) ≤
      2 * max initial (2 * Nat.nth Nat.Prime k) := by
  have total := attemptLimit_sum initial retries
  have last := firstSuccess_limit run
  omega

/-- Numeric envelope for the total attempted sieve sizes. -/
theorem source_total_limits (run : FirstSuccess k (initialLimit (k + 1)) retries) :
    (∑ j ∈ Finset.range (retries + 1), attemptLimit (initialLimit (k + 1)) j) ≤
      2 * max (initialLimit (k + 1)) (2 ^ (k + 2)) := by
  apply (firstSuccess_total_limits run).trans
  apply Nat.mul_le_mul_left
  apply max_le_max_left
  have prime := nth_prime_le_two_pow_succ k
  calc
    2 * Nat.nth Nat.Prime k ≤ 2 * 2 ^ (k + 1) := Nat.mul_le_mul_left _ prime
    _ = 2 ^ (k + 2) := by ring

/-- A simple numeric bound on the integer starting estimate. -/
theorem initialLimit_numeric (n : Nat) : initialLimit n ≤ 2 * n * (n + 1) + 15 := by
  have first := Nat.log2_le_self n
  have second := Nat.log2_le_self (Nat.log2 n + 1)
  unfold initialLimit
  split
  · omega
  · have combined : Nat.log2 n + Nat.log2 (Nat.log2 n + 1) + 1 ≤ 2 * (n + 1) := by omega
    have product := Nat.mul_le_mul_left n combined
    nlinarith

/-- One uniform numeric ceiling for every attempt. -/
def numericLimit (n : Nat) : Nat := 2 * n * (n + 1) + 15 + 2 ^ (n + 1)

theorem source_limit_numeric (run : FirstSuccess k (initialLimit (k + 1)) retries) :
    attemptLimit (initialLimit (k + 1)) retries ≤ numericLimit (k + 1) := by
  apply (firstSuccess_limit run).trans
  apply max_le
  · have := initialLimit_numeric (k + 1)
    unfold numericLimit
    omega
  · have bound := Nat.mul_le_mul_left 2 (nth_prime_le_two_pow_succ k)
    have power : 2 * 2 ^ (k + 1) = 2 ^ (k + 1 + 1) := by ring
    rw [power] at bound
    unfold numericLimit
    omega

end solvers.«7».complexity
