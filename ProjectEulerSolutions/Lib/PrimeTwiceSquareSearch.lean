import ProjectEulerSolutions.Lib.PrimeTwiceSquare

/-! Kernel-checked finite bounds for the first odd composite without a prime-plus-twice-square representation. -/
namespace ProjectEulerSolutions.PrimeTwiceSquareSearch
open PrimeTwiceSquare

set_option Elab.async false

def winner : Nat := 5777

private abbrev segment (first count : Nat) : Prop :=
  (List.range' first count 2).all (fun n => SixStepPrimality.isPrime n || certifiedFast n) = true

private theorem segment_append (first left right : Nat)
    (before : segment first left) (after : segment (first+2*left) right) : segment first (left+right) := by
  unfold segment
  rw [← List.range'_append]
  simp only [List.all_append, Bool.and_eq_true]
  exact ⟨before, after⟩

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_0 : segment 9 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_1 : segment 137 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_2 : segment 265 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_3 : segment 393 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_4 : segment 521 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_5 : segment 649 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_6 : segment 777 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_7 : segment 905 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_8 : segment 1033 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_9 : segment 1161 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_10 : segment 1289 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_11 : segment 1417 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_12 : segment 1545 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_13 : segment 1673 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_14 : segment 1801 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_15 : segment 1929 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_16 : segment 2057 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_17 : segment 2185 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_18 : segment 2313 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_19 : segment 2441 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_20 : segment 2569 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_21 : segment 2697 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_22 : segment 2825 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_23 : segment 2953 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_24 : segment 3081 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_25 : segment 3209 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_26 : segment 3337 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_27 : segment 3465 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_28 : segment 3593 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_29 : segment 3721 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_30 : segment 3849 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_31 : segment 3977 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_32 : segment 4105 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_33 : segment 4233 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_34 : segment 4361 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_35 : segment 4489 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_36 : segment 4617 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_37 : segment 4745 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_38 : segment 4873 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_39 : segment 5001 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_40 : segment 5129 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_41 : segment 5257 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_42 : segment 5385 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_43 : segment 5513 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_44 : segment 5641 64 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_45 : segment 5769 4 := by
  decide +kernel

private theorem rejected : segment 9 2884 := by
  have tail_45 : segment 5769 4 := segment_45
  have tail_44 : segment 5641 68 := segment_append 5641 64 4 segment_44 tail_45
  have tail_43 : segment 5513 132 := segment_append 5513 64 68 segment_43 tail_44
  have tail_42 : segment 5385 196 := segment_append 5385 64 132 segment_42 tail_43
  have tail_41 : segment 5257 260 := segment_append 5257 64 196 segment_41 tail_42
  have tail_40 : segment 5129 324 := segment_append 5129 64 260 segment_40 tail_41
  have tail_39 : segment 5001 388 := segment_append 5001 64 324 segment_39 tail_40
  have tail_38 : segment 4873 452 := segment_append 4873 64 388 segment_38 tail_39
  have tail_37 : segment 4745 516 := segment_append 4745 64 452 segment_37 tail_38
  have tail_36 : segment 4617 580 := segment_append 4617 64 516 segment_36 tail_37
  have tail_35 : segment 4489 644 := segment_append 4489 64 580 segment_35 tail_36
  have tail_34 : segment 4361 708 := segment_append 4361 64 644 segment_34 tail_35
  have tail_33 : segment 4233 772 := segment_append 4233 64 708 segment_33 tail_34
  have tail_32 : segment 4105 836 := segment_append 4105 64 772 segment_32 tail_33
  have tail_31 : segment 3977 900 := segment_append 3977 64 836 segment_31 tail_32
  have tail_30 : segment 3849 964 := segment_append 3849 64 900 segment_30 tail_31
  have tail_29 : segment 3721 1028 := segment_append 3721 64 964 segment_29 tail_30
  have tail_28 : segment 3593 1092 := segment_append 3593 64 1028 segment_28 tail_29
  have tail_27 : segment 3465 1156 := segment_append 3465 64 1092 segment_27 tail_28
  have tail_26 : segment 3337 1220 := segment_append 3337 64 1156 segment_26 tail_27
  have tail_25 : segment 3209 1284 := segment_append 3209 64 1220 segment_25 tail_26
  have tail_24 : segment 3081 1348 := segment_append 3081 64 1284 segment_24 tail_25
  have tail_23 : segment 2953 1412 := segment_append 2953 64 1348 segment_23 tail_24
  have tail_22 : segment 2825 1476 := segment_append 2825 64 1412 segment_22 tail_23
  have tail_21 : segment 2697 1540 := segment_append 2697 64 1476 segment_21 tail_22
  have tail_20 : segment 2569 1604 := segment_append 2569 64 1540 segment_20 tail_21
  have tail_19 : segment 2441 1668 := segment_append 2441 64 1604 segment_19 tail_20
  have tail_18 : segment 2313 1732 := segment_append 2313 64 1668 segment_18 tail_19
  have tail_17 : segment 2185 1796 := segment_append 2185 64 1732 segment_17 tail_18
  have tail_16 : segment 2057 1860 := segment_append 2057 64 1796 segment_16 tail_17
  have tail_15 : segment 1929 1924 := segment_append 1929 64 1860 segment_15 tail_16
  have tail_14 : segment 1801 1988 := segment_append 1801 64 1924 segment_14 tail_15
  have tail_13 : segment 1673 2052 := segment_append 1673 64 1988 segment_13 tail_14
  have tail_12 : segment 1545 2116 := segment_append 1545 64 2052 segment_12 tail_13
  have tail_11 : segment 1417 2180 := segment_append 1417 64 2116 segment_11 tail_12
  have tail_10 : segment 1289 2244 := segment_append 1289 64 2180 segment_10 tail_11
  have tail_9 : segment 1161 2308 := segment_append 1161 64 2244 segment_9 tail_10
  have tail_8 : segment 1033 2372 := segment_append 1033 64 2308 segment_8 tail_9
  have tail_7 : segment 905 2436 := segment_append 905 64 2372 segment_7 tail_8
  have tail_6 : segment 777 2500 := segment_append 777 64 2436 segment_6 tail_7
  have tail_5 : segment 649 2564 := segment_append 649 64 2500 segment_5 tail_6
  have tail_4 : segment 521 2628 := segment_append 521 64 2564 segment_4 tail_5
  have tail_3 : segment 393 2692 := segment_append 393 64 2628 segment_3 tail_4
  have tail_2 : segment 265 2756 := segment_append 265 64 2692 segment_2 tail_3
  have tail_1 : segment 137 2820 := segment_append 137 64 2756 segment_1 tail_2
  have tail_0 : segment 9 2884 := segment_append 9 64 2820 segment_0 tail_1
  exact tail_0

/-- The stride-two solver cannot return before the certified counterexample. -/
theorem no_earlier_odd (n : Nat) (lower : 9 ≤ n) (upper : n < winner) (odd : n%2=1) :
    counterexample n=false := by
  have member : n ∈ List.range' 9 2884 2 := by
    rw [List.mem_range']
    exact ⟨(n-9)/2, by dsimp [winner] at upper; omega, by omega⟩
  have tested := List.all_eq_true.mp rejected n member
  rw [SixStepPrimality.isPrime_correct, certifiedFast_correct] at tested
  simp only [counterexample, oddComposite]
  cases prime : decide (Nat.Prime n) <;> cases written : fast n <;> simp_all

private theorem small_checked :
    (List.range 9).all (fun n => !(decide (n%2=1 ∧ 1<n) &&
      !SixStepPrimality.isPrime n && !certifiedFast n))=true := by
  decide +kernel

/-- Even numbers and the small prefix cannot add an earlier reference result. -/
theorem no_earlier (n : Nat) (upper : n<winner) : counterexample n=false := by
  by_cases lower : 9≤n
  · by_cases odd : n%2=1
    · exact no_earlier_odd n lower upper odd
    · simp [counterexample, oddComposite, odd]
  · have tested := List.all_eq_true.mp small_checked n (List.mem_range.mpr (by omega))
    rw [SixStepPrimality.isPrime_correct, certifiedFast_correct] at tested
    simp only [counterexample, oddComposite]
    cases odd : decide (n%2=1) <;> cases large : decide (1<n) <;>
      cases prime : decide (Nat.Prime n) <;> cases written : fast n <;> simp_all

/-- This witness also proves termination of the unbounded optimized search. -/
theorem winner_accepted : counterexample winner=true := by
  have tested : (SixStepPrimality.isPrime winner, certifiedFast winner)=(false, false) := by
    decide +kernel
  rw [SixStepPrimality.isPrime_correct, certifiedFast_correct] at tested
  have composite : ¬ Nat.Prime winner := by
    exact of_decide_eq_false (Prod.mk.inj tested).1
  simp only [counterexample, oddComposite, composite, not_false_eq_true, and_true,
    show winner%2=1 by decide, show 1<winner by decide, decide_true,
    (Prod.mk.inj tested).2, Bool.not_false, Bool.and_self]

/-- The reference's fixed window contains the globally least counterexample. -/
theorem reference_find : (List.range 10000).find? counterexample=some winner := by
  apply List.find?_eq_some_iff_append.mpr
  refine ⟨winner_accepted, List.range' 0 5777, List.range' 5778 4222, ?_, ?_⟩
  · rw [List.range_eq_range']
    rw [← List.range'_append (s := 0) (m := 5777) (n := 4223) (step := 1)]
    congr 1
  · intro n member
    obtain ⟨offset, bound, equal⟩ := List.mem_range'.mp member
    have upper : n<winner := by dsimp [winner]; omega
    simp only [no_earlier n upper, Bool.not_false]

end ProjectEulerSolutions.PrimeTwiceSquareSearch
