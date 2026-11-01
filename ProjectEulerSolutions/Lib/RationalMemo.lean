import ProjectEulerSolutions.Lib.RationalKeys
import lang.py.FormalSW.KeyDictionaryCalls

/-! Typed reduction-state memo entries retain the first tuple key representative. -/
namespace ProjectEulerSolutions.RationalPair.Memo
open FormalSW.PyAst
set_option Elab.async false

structure Entry where
  state : List Pair
  stateAddress : Nat
  result : Nat

def Entry.encode (entry : Entry) : HashDict.Entry Mutable.Value :=
  (Heap.stateKey entry.state, .ref entry.stateAddress, .ref entry.result)

def encode (entries : List Entry) : List (HashDict.Entry Mutable.Value) := entries.map Entry.encode

def lookup : List Entry → List Pair → Option Nat
  | [], _ => none
  | entry :: rest, state => if entry.state = state then some entry.result else lookup rest state

def insert : List Entry → List Pair → Nat → Nat → List Entry
  | [], state, address, result => [⟨state, address, result⟩]
  | entry :: rest, state, address, result =>
      if entry.state = state then { entry with result } :: rest else entry :: insert rest state address result

theorem lookup_encode (entries : List Entry) (state : List Pair) :
    HashDict.lookup (encode entries) (Heap.stateKey state) = (lookup entries state).map Mutable.Value.ref := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
    simp only [encode, List.map_cons, HashDict.lookup_cons, Entry.encode, Heap.stateKey_equal]
    by_cases same : entry.state = state
    · simp [lookup, same]
    · simpa [encode, lookup, same] using ih

theorem insert_encode (entries : List Entry) (state : List Pair) (address result : Nat) :
    HashDict.insert (encode entries) (Heap.stateKey state) (.ref address) (.ref result) =
      encode (insert entries state address result) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
    simp only [encode, List.map_cons, HashDict.insert, Entry.encode, Heap.stateKey_equal]
    by_cases same : entry.state = state
    · simp [insert, same, Entry.encode]
    · simpa [insert, same, encode, Entry.encode] using congrArg (Entry.encode entry :: ·) ih

theorem lookup_insert (entries : List Entry) (state : List Pair) (address result : Nat) :
    lookup (insert entries state address result) state = some result := by
  induction entries with
  | nil => simp [insert, lookup]
  | cons entry rest ih =>
    by_cases same : entry.state = state <;> simp [insert, lookup, same, ih]

def Correct (relation : List Pair → Nat → Prop) (entries : List Entry) : Prop :=
  ∀ entry ∈ entries, relation entry.state entry.result

theorem correct_nil (relation : List Pair → Nat → Prop) : Correct relation [] := by
  intro entry member
  simp at member

theorem Correct.mono {relation next : List Pair → Nat → Prop} {entries : List Entry}
    (correct : Correct relation entries) (implication : ∀ state result, relation state result → next state result) :
    Correct next entries := fun entry member => implication entry.state entry.result (correct entry member)

theorem lookup_member {entries : List Entry} {state : List Pair} {result : Nat}
    (found : lookup entries state = some result) :
    ∃ entry ∈ entries, entry.state = state ∧ entry.result = result := by
  induction entries with
  | nil => simp [lookup] at found
  | cons entry rest ih =>
    by_cases same : entry.state = state
    · exact ⟨entry, by simp, same, by simpa [lookup, same] using found⟩
    · obtain ⟨old, member, stateEqual, resultEqual⟩ := ih (by simpa [lookup, same] using found)
      exact ⟨old, List.mem_cons_of_mem entry member, stateEqual, resultEqual⟩

theorem lookup_correct {relation : List Pair → Nat → Prop} {entries : List Entry} {state : List Pair} {result : Nat}
    (correct : Correct relation entries) (found : lookup entries state = some result) : relation state result := by
  obtain ⟨entry, member, rfl, rfl⟩ := lookup_member found
  exact correct entry member

theorem insert_correct {relation : List Pair → Nat → Prop} {entries : List Entry}
    {state : List Pair} {address result : Nat} (correct : Correct relation entries) (value : relation state result) :
    Correct relation (insert entries state address result) := by
  induction entries with
  | nil =>
    intro entry member
    obtain rfl := List.mem_singleton.mp member
    exact value
  | cons entry rest ih =>
    by_cases same : entry.state = state
    · intro item member
      simp only [insert, if_pos same, List.mem_cons] at member
      rcases member with rfl | member
      · simpa only [same] using value
      · exact correct item (List.mem_cons_of_mem entry member)
    · intro item member
      simp only [insert, if_neg same, List.mem_cons] at member
      rcases member with rfl | member
      · exact correct item (List.mem_cons_self ..)
      · exact ih (fun old present => correct old (List.mem_cons_of_mem entry present)) item member

/-- Every final cached set was already cached at entry or was allocated during this call. -/
def Origins (before : List Entry) (boundary : Nat) (after : List Entry) : Prop :=
  Correct (fun _ result => result ∈ before.map Entry.result ∨ boundary ≤ result) after

theorem Origins.refl (entries : List Entry) (boundary : Nat) : Origins entries boundary entries := by
  intro entry member
  exact .inl (List.mem_map.mpr ⟨entry, member, rfl⟩)

theorem Origins.insert {before after : List Entry} {boundary : Nat} (origins : Origins before boundary after)
    (state : List Pair) (address result : Nat) (origin : result ∈ before.map Entry.result ∨ boundary ≤ result) :
    Origins before boundary (insert after state address result) := insert_correct origins origin

theorem Origins.trans {before middle after : List Entry} {first second : Nat}
    (earlier : Origins before first middle) (later : Origins middle second after) (size : first ≤ second) :
    Origins before first after := by
  intro entry member
  rcases later entry member with old | fresh
  · obtain ⟨previous, present, equal⟩ := List.mem_map.mp old
    simpa only [equal] using earlier previous present
  · exact .inr (Nat.le_trans size fresh)

/-- A pre-existing private collector cannot become a cached result during a nested call. -/
theorem Origins.avoids {before after : List Entry} {boundary address : Nat}
    (origins : Origins before boundary after) (uncached : address ∉ before.map Entry.result) (old : address < boundary) :
    address ∉ after.map Entry.result := by
  intro member
  obtain ⟨entry, present, equal⟩ := List.mem_map.mp member
  rcases origins entry present with cached | fresh
  · exact uncached (equal ▸ cached)
  · omega

end ProjectEulerSolutions.RationalPair.Memo
