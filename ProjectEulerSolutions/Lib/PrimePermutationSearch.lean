import ProjectEulerSolutions.Lib.PrimePermutationGroups
import Mathlib.Data.List.TakeDrop

/-! Small kernel certificates classify all four-digit prime permutation progressions. -/
namespace ProjectEulerSolutions.PrimePermutationSearch
open PrimePermutationGroups

set_option Elab.async false

private abbrev segment (first count : Nat) : Prop :=
  ((keys.drop first).take count).all checkedKey=true

private theorem segment_append (first left right : Nat)
    (before : segment first left) (after : segment (first+left) right) : segment first (left+right) := by
  unfold segment at *
  rw [List.take_add, List.drop_drop, List.all_append, Bool.and_eq_true]
  exact ⟨before, after⟩

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_0 : segment 0 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_1 : segment 16 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_2 : segment 32 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_3 : segment 48 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_4 : segment 64 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_5 : segment 80 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_6 : segment 96 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_7 : segment 112 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_8 : segment 128 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_9 : segment 144 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_10 : segment 160 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_11 : segment 176 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_12 : segment 192 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_13 : segment 208 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_14 : segment 224 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_15 : segment 240 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_16 : segment 256 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_17 : segment 272 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_18 : segment 288 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_19 : segment 304 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_20 : segment 320 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_21 : segment 336 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_22 : segment 352 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_23 : segment 368 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_24 : segment 384 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_25 : segment 400 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_26 : segment 416 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_27 : segment 432 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_28 : segment 448 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_29 : segment 464 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_30 : segment 480 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_31 : segment 496 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_32 : segment 512 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_33 : segment 528 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_34 : segment 544 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_35 : segment 560 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_36 : segment 576 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_37 : segment 592 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_38 : segment 608 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_39 : segment 624 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_40 : segment 640 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_41 : segment 656 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_42 : segment 672 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_43 : segment 688 16 := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem segment_44 : segment 704 11 := by
  decide +kernel

private theorem all_checked : segment 0 715 := by
  have tail_44 : segment 704 11 := segment_44
  have tail_43 : segment 688 27 := segment_append 688 16 11 segment_43 tail_44
  have tail_42 : segment 672 43 := segment_append 672 16 27 segment_42 tail_43
  have tail_41 : segment 656 59 := segment_append 656 16 43 segment_41 tail_42
  have tail_40 : segment 640 75 := segment_append 640 16 59 segment_40 tail_41
  have tail_39 : segment 624 91 := segment_append 624 16 75 segment_39 tail_40
  have tail_38 : segment 608 107 := segment_append 608 16 91 segment_38 tail_39
  have tail_37 : segment 592 123 := segment_append 592 16 107 segment_37 tail_38
  have tail_36 : segment 576 139 := segment_append 576 16 123 segment_36 tail_37
  have tail_35 : segment 560 155 := segment_append 560 16 139 segment_35 tail_36
  have tail_34 : segment 544 171 := segment_append 544 16 155 segment_34 tail_35
  have tail_33 : segment 528 187 := segment_append 528 16 171 segment_33 tail_34
  have tail_32 : segment 512 203 := segment_append 512 16 187 segment_32 tail_33
  have tail_31 : segment 496 219 := segment_append 496 16 203 segment_31 tail_32
  have tail_30 : segment 480 235 := segment_append 480 16 219 segment_30 tail_31
  have tail_29 : segment 464 251 := segment_append 464 16 235 segment_29 tail_30
  have tail_28 : segment 448 267 := segment_append 448 16 251 segment_28 tail_29
  have tail_27 : segment 432 283 := segment_append 432 16 267 segment_27 tail_28
  have tail_26 : segment 416 299 := segment_append 416 16 283 segment_26 tail_27
  have tail_25 : segment 400 315 := segment_append 400 16 299 segment_25 tail_26
  have tail_24 : segment 384 331 := segment_append 384 16 315 segment_24 tail_25
  have tail_23 : segment 368 347 := segment_append 368 16 331 segment_23 tail_24
  have tail_22 : segment 352 363 := segment_append 352 16 347 segment_22 tail_23
  have tail_21 : segment 336 379 := segment_append 336 16 363 segment_21 tail_22
  have tail_20 : segment 320 395 := segment_append 320 16 379 segment_20 tail_21
  have tail_19 : segment 304 411 := segment_append 304 16 395 segment_19 tail_20
  have tail_18 : segment 288 427 := segment_append 288 16 411 segment_18 tail_19
  have tail_17 : segment 272 443 := segment_append 272 16 427 segment_17 tail_18
  have tail_16 : segment 256 459 := segment_append 256 16 443 segment_16 tail_17
  have tail_15 : segment 240 475 := segment_append 240 16 459 segment_15 tail_16
  have tail_14 : segment 224 491 := segment_append 224 16 475 segment_14 tail_15
  have tail_13 : segment 208 507 := segment_append 208 16 491 segment_13 tail_14
  have tail_12 : segment 192 523 := segment_append 192 16 507 segment_12 tail_13
  have tail_11 : segment 176 539 := segment_append 176 16 523 segment_11 tail_12
  have tail_10 : segment 160 555 := segment_append 160 16 539 segment_10 tail_11
  have tail_9 : segment 144 571 := segment_append 144 16 555 segment_9 tail_10
  have tail_8 : segment 128 587 := segment_append 128 16 571 segment_8 tail_9
  have tail_7 : segment 112 603 := segment_append 112 16 587 segment_7 tail_8
  have tail_6 : segment 96 619 := segment_append 96 16 603 segment_6 tail_7
  have tail_5 : segment 80 635 := segment_append 80 16 619 segment_5 tail_6
  have tail_4 : segment 64 651 := segment_append 64 16 635 segment_4 tail_5
  have tail_3 : segment 48 667 := segment_append 48 16 651 segment_3 tail_4
  have tail_2 : segment 32 683 := segment_append 32 16 667 segment_2 tail_3
  have tail_1 : segment 16 699 := segment_append 16 16 683 segment_1 tail_2
  have tail_0 : segment 0 715 := segment_append 0 16 699 segment_0 tail_1
  exact tail_0

/-- Every canonical four-digit key has been checked, independently of its traversal order. -/
theorem checked (key : List Nat) (member : key ∈ keys) : checkedKey key=true := by
  have length : keys.length=715 := by decide +kernel
  have whole : keys.all checkedKey=true := by
    simpa only [segment, List.drop_zero, ← length, List.take_length] using all_checked
  exact List.all_eq_true.mp whole key member

/-- Every increasing four-digit prime arithmetic triple is one of the two certified rows. -/
theorem classifies (a b c : Nat) (lower : 1000≤a) (upper : c<10000)
    (ab : a<b) (bc : b<c) (arithmetic : c=2*b-a)
    (aPrime : Nat.Prime a) (bPrime : Nat.Prime b) (cPrime : Nat.Prime c)
    (bDigits : ProjectEulerStatements.P49.isPermutation a b=true)
    (cDigits : ProjectEulerStatements.P49.isPermutation a c=true) :
    (a,b,c)=known ∨ (a,b,c)=winner := by
  let key := ProjectEulerStatements.P49.digitsSorted a
  have aPermutation : (Nat.digits 10 a).Perm key := (List.mergeSort_perm _ _).symm
  have bPermutation : (Nat.digits 10 b).Perm key := by
    have same : ProjectEulerStatements.P49.digitsSorted a=ProjectEulerStatements.P49.digitsSorted b :=
      of_decide_eq_true bDigits
    dsimp [key]
    rw [same]
    exact (List.mergeSort_perm _ _).symm
  have cPermutation : (Nat.digits 10 c).Perm key := by
    have same : ProjectEulerStatements.P49.digitsSorted a=ProjectEulerStatements.P49.digitsSorted c :=
      of_decide_eq_true cDigits
    dsimp [key]
    rw [same]
    exact (List.mergeSort_perm _ _).symm
  exact checkedKey_classifies key (checked key (canonical_key_mem a lower (by omega))) a b c
    (group_mem key a lower (by omega) aPrime aPermutation)
    (group_mem key b (by omega) (by omega) bPrime bPermutation)
    (group_mem key c (by omega) upper cPrime cPermutation) ab arithmetic

end ProjectEulerSolutions.PrimePermutationSearch
