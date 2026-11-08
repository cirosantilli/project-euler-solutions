import solvers.«84.py.lean.d».LandingWeightDistinct

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst
set_option Elab.async false

def Positive (entries : List Entry) : Prop := ∀ entry ∈ entries, 0 < entry.weight

theorem lookup_member (entries : List Entry) (landing : Int × Bool) (weight : Int)
    (found : lookup entries landing = some weight) :
    ∃ entry ∈ entries, entry.landing = landing ∧ entry.weight = weight := by
  induction entries with
  | nil => simp [lookup] at found
  | cons entry rest ih =>
      by_cases same : entry.landing = landing
      · exact ⟨entry, by simp, same, by simpa [lookup, same] using found⟩
      · obtain ⟨current, member, key, value⟩ := ih (by simpa [lookup, same] using found)
        exact ⟨current, by simp [member], key, value⟩

theorem lookup_nonneg (entries : List Entry) (positive : Positive entries) (landing : Int × Bool) :
    0 ≤ (lookup entries landing).getD 0 := by
  cases found : lookup entries landing with
  | none => simp
  | some weight =>
      obtain ⟨entry, member, _, same⟩ := lookup_member entries landing weight found
      have pos := positive entry member
      simp only [Option.getD_some]
      omega

theorem Positive.inserted {entries : List Entry} (positive : Positive entries)
    (landing : Int × Bool) (address : Nat) (weight : Int) (pos : 0 < weight) :
    Positive (insert entries landing address weight) := by
  induction entries with
  | nil => simpa [Positive, insert] using pos
  | cons entry rest ih =>
      have tail : Positive rest := fun current member => positive current (by simp [member])
      intro current member
      by_cases same : entry.landing = landing
      · simp only [insert, if_pos same, List.mem_cons] at member
        rcases member with rfl | member
        · exact pos
        · exact tail current member
      · simp only [insert, if_neg same, List.mem_cons] at member
        rcases member with rfl | member
        · exact positive current (by simp)
        · exact ih tail current member

theorem Positive.added {entries : List Entry} (positive : Positive entries)
    (landing : Int × Bool) (address : Nat) (amount : Int) (pos : 0 < amount) :
    Positive (add entries landing address amount) := by
  apply positive.inserted
  have previous := lookup_nonneg entries positive landing
  omega

theorem lookup_at (entries : List Entry) (entry : Entry) (member : entry ∈ entries) (distinct : Distinct entries) :
    lookup entries entry.landing = some entry.weight := by
  induction entries with
  | nil => simp at member
  | cons head rest ih =>
      have separate := List.nodup_cons.mp distinct
      rcases List.mem_cons.mp member with rfl | member
      · simp [lookup]
      · have different : head.landing ≠ entry.landing := by
          intro same
          apply separate.1
          rw [same]
          exact List.mem_map.mpr ⟨entry, member, rfl⟩
        simp [lookup, different, ih member separate.2]

end solvers.«84».resolve_landing.weights
