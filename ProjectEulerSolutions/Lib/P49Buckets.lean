import ProjectEulerSolutions.Lib.PrimePermutationReference

/-! Four-digit prime permutation buckets, preserving the exhaustive reference's row order. -/
namespace ProjectEulerSolutions.P49Buckets
open PrimePermutationGroups
set_option Elab.async false

/-- A fixed-width pair key sorts first by the initial prime and then by its successor. -/
def encode (a b : Nat) : Nat := a * 10000 + b

def pairs (values : List Nat) : List Nat :=
  values.flatMap fun a => values.filterMap fun b =>
    if a < b ∧ 2*b-a ∈ values then some (encode a b) else none

/-- Generate each canonical digit-multiset bucket once; primality is checked only for its permutations. -/
def codes : List Nat := keys.flatMap fun key => pairs (group key)

def decode (code : Nat) : List Nat :=
  let a := code / 10000
  let b := code % 10000
  [a, b, 2*b-a]

def rows : List (List Nat) :=
  (codes.dedup.mergeSort (fun a b => decide (a ≤ b))).map decode

theorem mem_pairs (values : List Nat) (code : Nat) : code ∈ pairs values ↔
    ∃ a ∈ values, ∃ b ∈ values, a < b ∧ 2*b-a ∈ values ∧ code = encode a b := by
  simp only [pairs, List.mem_flatMap, List.mem_filterMap]
  constructor
  · rintro ⟨a, ha, b, hb, selected⟩
    split at selected
    · rename_i accepted
      exact ⟨a, ha, b, hb, accepted.1, accepted.2, (Option.some.inj selected).symm⟩
    · contradiction
  · rintro ⟨a, ha, b, hb, ab, hc, rfl⟩
    exact ⟨a, ha, b, hb, by simp [ab, hc]⟩

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem known_present : encode 1487 4817 ∈ pairs (group [1,4,7,8]) := by
  decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
private theorem winner_present : encode 2969 6299 ∈ pairs (group [2,6,9,9]) := by
  decide +kernel

/-- The existing classification certificates exclude every other bucket result. -/
theorem mem_codes (code : Nat) : code ∈ codes ↔ code = encode 1487 4817 ∨ code = encode 2969 6299 := by
  constructor
  · intro member
    obtain ⟨key, keyMember, pairMember⟩ := List.mem_flatMap.mp member
    obtain ⟨a, ha, b, hb, ab, hc, rfl⟩ := (mem_pairs _ _).mp pairMember
    have classification := checkedKey_classifies key (PrimePermutationSearch.checked key keyMember)
      a b (2*b-a) ha hb hc ab rfl
    rcases classification with knownRow | winningRow
    · left
      simpa only [known] using congrArg (fun row : Row => encode row.1 row.2.1) knownRow
    · right
      simpa only [winner] using congrArg (fun row : Row => encode row.1 row.2.1) winningRow
  · rintro (rfl | rfl)
    · exact List.mem_flatMap.mpr ⟨[1,4,7,8], key_mem _ _ _ _ (by decide) (by decide) (by decide) (by decide), known_present⟩
    · exact List.mem_flatMap.mpr ⟨[2,6,9,9], key_mem _ _ _ _ (by decide) (by decide) (by decide) (by decide), winner_present⟩

theorem ordered_codes : codes.dedup.mergeSort (fun a b => decide (a ≤ b)) =
    [encode 1487 4817, encode 2969 6299] := by
  have permutation : codes.dedup.Perm [encode 1487 4817, encode 2969 6299] := by
    apply (List.perm_ext_iff_of_nodup (List.nodup_dedup _) (by decide)).mpr
    intro code
    simp only [List.mem_dedup, List.mem_cons, List.not_mem_nil, or_false, mem_codes]
  have sorted := List.pairwise_mergeSort' (· ≤ ·) codes.dedup
  exact ((List.mergeSort_perm _ _).trans permutation).eq_of_pairwise' sorted (by decide)

theorem rows_correct : rows = ProjectEulerStatements.P49.naive 4 3 := by
  rw [PrimePermutationReference.reference_rows, rows, ordered_codes]
  rfl
end ProjectEulerSolutions.P49Buckets
