import ProjectEulerSolutions.Lib.PrimePermutations
import ProjectEulerSolutions.Lib.SixStepPrimality
import Mathlib.Data.List.Permutation

/-! Finite digit-multiset groups cover every four-digit prime permutation triple. -/
namespace ProjectEulerSolutions.PrimePermutationGroups
open PrimePermutations

set_option Elab.async false

abbrev Row := Nat×Nat×Nat
def known : Row := (1487,4817,8147)
def winner : Row := (2969,6299,9629)

def keys : List (List Nat) := (List.range 10).flatMap fun a =>
  (List.range' a (10-a)).flatMap fun b =>
    (List.range' b (10-b)).flatMap fun c =>
      (List.range' c (10-c)).map fun d => [a,b,c,d]

def group (key : List Nat) : List Nat :=
  ((key.permutations'.map (Nat.ofDigits 10)).dedup).filter fun n =>
    decide (1000≤n ∧ n<10000) && SixStepPrimality.certifiedPrime n

/-- Only pairs in the same small digit group need to be checked. -/
def checkedKey (key : List Nat) : Bool :=
  let values := group key
  values.all fun a => values.all fun b =>
    if a<b ∧ 2*b-a ∈ values then decide ((a,b,2*b-a)=known ∨ (a,b,2*b-a)=winner) else true

theorem key_mem (a b c d : Nat) (ab : a≤b) (bc : b≤c) (cd : c≤d) (upper : d<10) :
    [a,b,c,d] ∈ keys := by
  simp only [keys, List.mem_flatMap, List.mem_range, List.mem_map, List.mem_range'_1]
  exact ⟨a, by omega, b, ⟨ab, by omega⟩, c, ⟨bc, by omega⟩, d, ⟨cd, by omega⟩, rfl⟩

/-- Sorting the four decimal digits picks one of the 715 nondecreasing keys. -/
theorem canonical_key_mem (n : Nat) (lower : 1000≤n) (upper : n<10000) :
    ProjectEulerStatements.P49.digitsSorted n ∈ keys := by
  let key := ProjectEulerStatements.P49.digitsSorted n
  have permutation : key.Perm (Nat.digits 10 n) := List.mergeSort_perm _ _
  have four : (Nat.digits 10 n).length=4 := by
    have atMost : (Nat.digits 10 n).length≤4 :=
      (Nat.digits_length_le_iff (by decide) n).mpr upper
    have atLeast : 3<(Nat.digits 10 n).length :=
      (Nat.lt_digits_length_iff (by decide) n).mpr lower
    omega
  have keyLength : key.length=4 := permutation.length_eq.trans four
  obtain ⟨a,b,c,d,equal⟩ := List.length_eq_four.mp keyLength
  have ordered : key.Pairwise (· ≤ ·) := List.pairwise_mergeSort' (· ≤ ·) (Nat.digits 10 n)
  rw [equal] at ordered
  have first := List.pairwise_cons.mp ordered
  have second := List.pairwise_cons.mp first.2
  have third := List.pairwise_cons.mp second.2
  have digitUpper : d<10 := Nat.digits_lt_base (by decide) (permutation.mem_iff.mp (by rw [equal]; simp))
  change key ∈ keys
  rw [equal]
  exact key_mem a b c d (first.1 b (by simp)) (second.1 c (by simp)) (third.1 d (by simp)) digitUpper

/-- Every four-digit prime with the key's digit multiset occurs in its generated permutation group. -/
theorem group_mem (key : List Nat) (n : Nat) (lower : 1000≤n) (upper : n<10000)
    (prime : Nat.Prime n) (permutation : (Nat.digits 10 n).Perm key) : n ∈ group key := by
  apply List.mem_filter.mpr
  refine ⟨?_, ?_⟩
  · rw [List.mem_dedup]
    exact List.mem_map.mpr ⟨Nat.digits 10 n, List.mem_permutations'.mpr permutation, Nat.ofDigits_digits 10 n⟩
  · simp [SixStepPrimality.certifiedPrime_correct, lower, upper, prime]

/-- A checked key classifies every increasing arithmetic triple drawn from its members. -/
theorem checkedKey_classifies (key : List Nat) (checked : checkedKey key=true) (a b c : Nat)
    (aMember : a ∈ group key) (bMember : b ∈ group key) (cMember : c ∈ group key)
    (increasing : a<b) (arithmetic : c=2*b-a) : (a,b,c)=known ∨ (a,b,c)=winner := by
  have tested := List.all_eq_true.mp (List.all_eq_true.mp checked a aMember) b bMember
  have condition : a<b ∧ 2*b-a ∈ group key := ⟨increasing, by simpa only [arithmetic] using cMember⟩
  simp only [if_pos condition, decide_eq_true_eq] at tested
  simpa only [arithmetic] using tested

end ProjectEulerSolutions.PrimePermutationGroups
