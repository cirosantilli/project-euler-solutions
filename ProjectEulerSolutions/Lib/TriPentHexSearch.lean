import ProjectEulerSolutions.Lib.TriPentHex

/-! Small kernel certificates establish the first common value after the given example.
The exhaustive reference is compared mathematically; it is never executed on huge values. -/
namespace ProjectEulerSolutions.TriPentHexSearch
open TriPentHex

set_option Elab.async false

def firstHex : Nat := 144
def winner : Nat := 27693
def triangleIndex : Nat := 2*winner-1

private abbrev segment (first count : Nat) : Prop :=
  (List.range' first count).all (fun n => !hexTest n) = true

private theorem segment_append (first left right : Nat)
    (before : segment first left) (after : segment (first+left) right) : segment first (left+right) := by
  unfold segment
  rw [← List.range'_append]
  simp only [Nat.one_mul, List.all_append, Bool.and_eq_true]
  exact ⟨before, after⟩

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_0 : segment 144 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_1 : segment 656 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_2 : segment 1168 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_3 : segment 1680 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_4 : segment 2192 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_5 : segment 2704 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_6 : segment 3216 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_7 : segment 3728 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_8 : segment 4240 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_9 : segment 4752 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_10 : segment 5264 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_11 : segment 5776 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_12 : segment 6288 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_13 : segment 6800 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_14 : segment 7312 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_15 : segment 7824 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_16 : segment 8336 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_17 : segment 8848 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_18 : segment 9360 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_19 : segment 9872 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_20 : segment 10384 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_21 : segment 10896 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_22 : segment 11408 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_23 : segment 11920 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_24 : segment 12432 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_25 : segment 12944 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_26 : segment 13456 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_27 : segment 13968 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_28 : segment 14480 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_29 : segment 14992 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_30 : segment 15504 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_31 : segment 16016 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_32 : segment 16528 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_33 : segment 17040 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_34 : segment 17552 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_35 : segment 18064 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_36 : segment 18576 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_37 : segment 19088 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_38 : segment 19600 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_39 : segment 20112 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_40 : segment 20624 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_41 : segment 21136 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_42 : segment 21648 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_43 : segment 22160 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_44 : segment 22672 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_45 : segment 23184 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_46 : segment 23696 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_47 : segment 24208 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_48 : segment 24720 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_49 : segment 25232 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_50 : segment 25744 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_51 : segment 26256 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_52 : segment 26768 512 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_53 : segment 27280 413 := by
  decide +kernel

/-- All rejected indices are checked in bounded blocks, without asynchronous compiler tasks. -/
private theorem rejected : segment firstHex (winner-firstHex) := by
  have tail_53 : segment 27280 413 := segment_53
  have tail_52 : segment 26768 925 := segment_append 26768 512 413 segment_52 tail_53
  have tail_51 : segment 26256 1437 := segment_append 26256 512 925 segment_51 tail_52
  have tail_50 : segment 25744 1949 := segment_append 25744 512 1437 segment_50 tail_51
  have tail_49 : segment 25232 2461 := segment_append 25232 512 1949 segment_49 tail_50
  have tail_48 : segment 24720 2973 := segment_append 24720 512 2461 segment_48 tail_49
  have tail_47 : segment 24208 3485 := segment_append 24208 512 2973 segment_47 tail_48
  have tail_46 : segment 23696 3997 := segment_append 23696 512 3485 segment_46 tail_47
  have tail_45 : segment 23184 4509 := segment_append 23184 512 3997 segment_45 tail_46
  have tail_44 : segment 22672 5021 := segment_append 22672 512 4509 segment_44 tail_45
  have tail_43 : segment 22160 5533 := segment_append 22160 512 5021 segment_43 tail_44
  have tail_42 : segment 21648 6045 := segment_append 21648 512 5533 segment_42 tail_43
  have tail_41 : segment 21136 6557 := segment_append 21136 512 6045 segment_41 tail_42
  have tail_40 : segment 20624 7069 := segment_append 20624 512 6557 segment_40 tail_41
  have tail_39 : segment 20112 7581 := segment_append 20112 512 7069 segment_39 tail_40
  have tail_38 : segment 19600 8093 := segment_append 19600 512 7581 segment_38 tail_39
  have tail_37 : segment 19088 8605 := segment_append 19088 512 8093 segment_37 tail_38
  have tail_36 : segment 18576 9117 := segment_append 18576 512 8605 segment_36 tail_37
  have tail_35 : segment 18064 9629 := segment_append 18064 512 9117 segment_35 tail_36
  have tail_34 : segment 17552 10141 := segment_append 17552 512 9629 segment_34 tail_35
  have tail_33 : segment 17040 10653 := segment_append 17040 512 10141 segment_33 tail_34
  have tail_32 : segment 16528 11165 := segment_append 16528 512 10653 segment_32 tail_33
  have tail_31 : segment 16016 11677 := segment_append 16016 512 11165 segment_31 tail_32
  have tail_30 : segment 15504 12189 := segment_append 15504 512 11677 segment_30 tail_31
  have tail_29 : segment 14992 12701 := segment_append 14992 512 12189 segment_29 tail_30
  have tail_28 : segment 14480 13213 := segment_append 14480 512 12701 segment_28 tail_29
  have tail_27 : segment 13968 13725 := segment_append 13968 512 13213 segment_27 tail_28
  have tail_26 : segment 13456 14237 := segment_append 13456 512 13725 segment_26 tail_27
  have tail_25 : segment 12944 14749 := segment_append 12944 512 14237 segment_25 tail_26
  have tail_24 : segment 12432 15261 := segment_append 12432 512 14749 segment_24 tail_25
  have tail_23 : segment 11920 15773 := segment_append 11920 512 15261 segment_23 tail_24
  have tail_22 : segment 11408 16285 := segment_append 11408 512 15773 segment_22 tail_23
  have tail_21 : segment 10896 16797 := segment_append 10896 512 16285 segment_21 tail_22
  have tail_20 : segment 10384 17309 := segment_append 10384 512 16797 segment_20 tail_21
  have tail_19 : segment 9872 17821 := segment_append 9872 512 17309 segment_19 tail_20
  have tail_18 : segment 9360 18333 := segment_append 9360 512 17821 segment_18 tail_19
  have tail_17 : segment 8848 18845 := segment_append 8848 512 18333 segment_17 tail_18
  have tail_16 : segment 8336 19357 := segment_append 8336 512 18845 segment_16 tail_17
  have tail_15 : segment 7824 19869 := segment_append 7824 512 19357 segment_15 tail_16
  have tail_14 : segment 7312 20381 := segment_append 7312 512 19869 segment_14 tail_15
  have tail_13 : segment 6800 20893 := segment_append 6800 512 20381 segment_13 tail_14
  have tail_12 : segment 6288 21405 := segment_append 6288 512 20893 segment_12 tail_13
  have tail_11 : segment 5776 21917 := segment_append 5776 512 21405 segment_11 tail_12
  have tail_10 : segment 5264 22429 := segment_append 5264 512 21917 segment_10 tail_11
  have tail_9 : segment 4752 22941 := segment_append 4752 512 22429 segment_9 tail_10
  have tail_8 : segment 4240 23453 := segment_append 4240 512 22941 segment_8 tail_9
  have tail_7 : segment 3728 23965 := segment_append 3728 512 23453 segment_7 tail_8
  have tail_6 : segment 3216 24477 := segment_append 3216 512 23965 segment_6 tail_7
  have tail_5 : segment 2704 24989 := segment_append 2704 512 24477 segment_5 tail_6
  have tail_4 : segment 2192 25501 := segment_append 2192 512 24989 segment_4 tail_5
  have tail_3 : segment 1680 26013 := segment_append 1680 512 25501 segment_3 tail_4
  have tail_2 : segment 1168 26525 := segment_append 1168 512 26013 segment_2 tail_3
  have tail_1 : segment 656 27037 := segment_append 656 512 26525 segment_1 tail_2
  have tail_0 : segment 144 27549 := segment_append 144 512 27037 segment_0 tail_1
  exact tail_0

theorem no_earlier (n : Nat) (lower : firstHex ≤ n) (upper : n < winner) : hexTest n = false := by
  have member : n ∈ List.range' firstHex (winner-firstHex) := by
    rw [List.mem_range']
    exact ⟨n-firstHex, by dsimp [firstHex, winner] at *; omega, by simp; omega⟩
  have tested := List.all_eq_true.mp rejected n member
  cases value : hexTest n with
  | false => rfl
  | true => simp [value] at tested

/-- The final index is a witness, so the unbounded Python loop must return by this point. -/
theorem winner_accepted : hexTest winner = true := by
  decide +kernel

theorem common_winner : common triangleIndex = true := by
  have positive : 0<winner := by decide
  simp only [common, triangleIndex, hex_triangular winner positive, hex_witness, Bool.and_true]
  rw [pent_positive (hexagonal winner) (hex_positive winner positive)]
  exact winner_accepted

/-- A putative earlier reference match would give an earlier match in the certified hexagonal scan. -/
theorem common_rejected (index : Nat) (lower : 286 ≤ index) (upper : index < triangleIndex) : common index = false := by
  cases accepted : common index with
  | false => rfl
  | true =>
      obtain ⟨n, _, indices, tested⟩ := common_hex index (by omega) accepted
      have afterStart : firstHex ≤ n := by dsimp [firstHex]; omega
      have beforeEnd : n < winner := by dsimp [triangleIndex] at upper; omega
      have rejected := no_earlier n afterStart beforeEnd
      rw [rejected] at tested
      cases tested

/-- The bounded reference window contains the same first match as the unbounded hexagonal search. -/
theorem reference_find : (List.range' 286 100001).find? common = some triangleIndex := by
  apply List.find?_eq_some_iff_append.mpr
  refine ⟨common_winner, List.range' 286 55099, List.range' 55386 44901, ?_, ?_⟩
  · rw [← List.range'_append (s := 286) (m := 55099) (n := 44902) (step := 1)]
    congr 1
  · intro index member
    obtain ⟨offset, bound, equal⟩ := List.mem_range'.mp member
    have lower : 286 ≤ index := by omega
    have upper : index < triangleIndex := by dsimp [triangleIndex, winner]; omega
    simp only [common_rejected index lower upper, Bool.not_false]

end ProjectEulerSolutions.TriPentHexSearch
