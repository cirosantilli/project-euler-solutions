import ProjectEulerSolutions.Lib.TotientArrayScan

set_option linter.style.header false

/-! Moving-width and cached-totient invariants identify the array scan with the reference fold. -/
namespace ProjectEulerSolutions.TotientArrayScan
open ProjectEulerStatements TotientPermutation
set_option Elab.async false

def canonical (best bestPhi width : Nat) : State :=
  ⟨best, bestPhi, 10 ^ width, 10 ^ (width + 1)⟩

theorem histogram_permutation (a b : Nat) :
    decide (digitSignature a = digitSignature b) = isPermutation a b := by
  apply Bool.eq_iff_iff.mpr
  simp only [digitSignature, decide_eq_true_eq, DigitHistogram.signature_eq_iff, permutation_iff]

theorem boundary_canonical (n best bestPhi width : Nat) :
    boundary n (canonical best bestPhi width) = canonical best bestPhi (advanceWidth n width) := by
  by_cases atBoundary : n = 10 ^ (width + 1)
  · simp only [boundary, canonical, if_pos atBoundary, advanceWidth]
    simp only [pow_succ]
  · simp only [boundary, canonical, if_neg atBoundary, advanceWidth]

theorem evaluate_canonical (n width best bestPhi : Nat) :
    evaluate n n.totient (canonical best bestPhi width) =
      canonical (scanStep n width best bestPhi).1 (scanStep n width best bestPhi).2 width := by
  by_cases badWidth : n.totient < 10 ^ width ∨ 10 ^ (width + 1) ≤ n.totient
  · have notRange : ¬ (10 ^ width ≤ n.totient ∧ n.totient < 10 ^ (width + 1)) := by omega
    simp [evaluate, canonical, scanStep, screen, badWidth, notRange]
  · have widthGood : 10 ^ width ≤ n.totient ∧ n.totient < 10 ^ (width + 1) := by omega
    by_cases badResidue : ((n : Int) - n.totient) % 9 ≠ 0
    · simp [evaluate, canonical, scanStep, screen, badWidth, badResidue, widthGood]
    · have residue : ((n : Int) - n.totient) % 9 = 0 := by omega
      cases same : isPermutation n n.totient with
      | false =>
          have different : digitSignature n ≠ digitSignature n.totient := by
            intro equal
            have contradiction : isPermutation n n.totient = true := by
              rw [← histogram_permutation, equal]
              simp
            simp [same] at contradiction
          simp [evaluate, canonical, scanStep, screen, badWidth, residue, same, different]
      | true =>
          have identical : digitSignature n = digitSignature n.totient :=
            of_decide_eq_true (by rw [histogram_permutation]; exact same)
          by_cases improved : best = 0 ∨ n * bestPhi < best * n.totient <;>
            simp [evaluate, canonical, scanStep, screen, badWidth, residue, same,
              identical, widthGood, improved]

theorem step_canonical (phi : Array Nat) (n best bestPhi width : Nat)
    (read : phi[n]! = n.totient) :
    step phi n (canonical best bestPhi width) =
      canonical (scanStep n (advanceWidth n width) best bestPhi).1
        (scanStep n (advanceWidth n width) best bestPhi).2 (advanceWidth n width) := by
  rw [step, boundary_canonical, read]
  exact evaluate_canonical n (advanceWidth n width) best bestPhi

/-- The array scan implements the same strict candidate fold for every finite interval. -/
theorem scan_correct (limit first count best bestPhi width : Nat) (phi : Array Nat)
    (length : first + count = limit + 1) (large : 2 ≤ first) (before : WidthBefore first width)
    (matched : PhiMatches best bestPhi) (correct : ∀ n, n ≤ limit → phi[n]! = n.totient) :
    scan limit first phi (canonical best bestPhi width) =
      (List.range' first count).foldl visit best := by
  induction count generalizing first best bestPhi width with
  | zero =>
      rw [scan, if_neg (by omega), List.range'_zero, List.foldl_nil]
      rfl
  | succ count ih =>
      rw [scan, if_pos (by omega),
        step_canonical phi first best bestPhi width (correct first (by omega))]
      let active := advanceWidth first width
      let selected := scanStep first active best bestPhi
      have nextBefore := advanceWidth_next first width before
      have nextMatched := scanStep_matches first active best bestPhi matched
      have continued := ih (first := first + 1) (best := selected.1) (bestPhi := selected.2)
        (width := active) (length := by omega) (large := by omega)
        (before := nextBefore) (matched := nextMatched)
      obtain ⟨lower, upper⟩ := advanceWidth_active first width before
      have chosen := scanStep_first first active best bestPhi large lower upper matched
      change scan limit (first + 1) phi (canonical selected.1 selected.2 active) =
        (List.range' first (count + 1)).foldl visit best
      have firstEq : selected.1 = visit best first := chosen
      simpa only [firstEq, List.range'_succ, List.foldl_cons] using continued

theorem solveCore_correct (limit : Nat) (large : 2 ≤ limit) :
    solveCore limit = result (limit : Int) := by
  have matched : PhiMatches 0 1 := fun nonzero => False.elim (nonzero rfl)
  have folded := scan_correct limit 2 (limit - 1) 0 1 0
    (TotientArraySieve.computePhiLinear limit) (by omega) (by decide)
    (by constructor <;> decide) matched (TotientArraySieve.computePhiLinear_correct limit)
  rw [fold_from_two limit (by omega)] at folded
  exact folded

/-- The standalone optimized model handles the same signed bounds as its Python source. -/
theorem solve_correct (limit : Int) : solve limit = result limit := by
  by_cases small : limit < 2
  · rw [solve, if_pos small, result_small limit small]
  · have cast : (limit.toNat : Int) = limit := Int.toNat_of_nonneg (by omega)
    rw [solve, if_neg small, solveCore_correct limit.toNat (by omega), cast]

end ProjectEulerSolutions.TotientArrayScan
