import solvers.«84.py.lean.d».LandingWeights

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst
set_option Elab.async false

def Distinct (entries : List Entry) : Prop := (entries.map Entry.landing).Nodup

/-- Updating an existing key keeps its position; inserting a new key appends it once. -/
theorem keys_insert (entries : List Entry) (landing : Int × Bool) (address : Nat) (weight : Int) :
    (insert entries landing address weight).map Entry.landing =
      if landing ∈ entries.map Entry.landing then entries.map Entry.landing
      else entries.map Entry.landing ++ [landing] := by
  induction entries with
  | nil => simp [insert]
  | cons entry rest ih =>
      by_cases same : entry.landing = landing
      · simp [insert, same]
      · by_cases present : landing ∈ rest.map Entry.landing <;>
          simp [insert, same, Ne.symm same, ih, present]

theorem Distinct.inserted {entries : List Entry} (distinct : Distinct entries)
    (landing : Int × Bool) (address : Nat) (weight : Int) : Distinct (insert entries landing address weight) := by
  unfold Distinct
  rw [keys_insert]
  split
  · exact distinct
  · rename_i absent
    apply List.nodup_append.mpr
    refine ⟨distinct, by simp, ?_⟩
    intro item member last only same
    have lastSame : last = landing := by simpa using only
    exact absent ((same.trans lastSame) ▸ member)

theorem Distinct.added {entries : List Entry} (distinct : Distinct entries)
    (landing : Int × Bool) (address : Nat) (weight : Int) : Distinct (add entries landing address weight) :=
  distinct.inserted landing address _

theorem lookup_absent (entries : List Entry) (landing : Int × Bool)
    (absent : landing ∉ entries.map Entry.landing) : lookup entries landing = none := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      have missing : landing ≠ entry.landing ∧ landing ∉ rest.map Entry.landing := by simpa using absent
      simp [lookup, Ne.symm missing.1, ih missing.2]

/-- Distinct dictionary keys let iteration sum the same weights as keyed lookup. -/
theorem lookup_sum (entries : List Entry) (landing : Int × Bool) (distinct : Distinct entries) :
    (lookup entries landing).getD 0 =
      (entries.map fun entry => if landing = entry.landing then entry.weight else 0).sum := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      have separated := List.nodup_cons.mp distinct
      by_cases same : entry.landing = landing
      · subst landing
        have missing := lookup_absent rest entry.landing separated.1
        have zero : (rest.map fun current => if entry.landing = current.landing then current.weight else 0).sum = 0 := by
          rw [← ih separated.2, missing]
          rfl
        simp [lookup, zero]
      · simp [lookup, same, Ne.symm same, ih separated.2]

end solvers.«84».resolve_landing.weights
