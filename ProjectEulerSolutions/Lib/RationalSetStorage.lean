import ProjectEulerSolutions.Lib.RationalSets
import lang.py.FormalSW.SetUpdateCalls

/-! Rational result sets retain their original tuple representatives during union updates. -/
namespace ProjectEulerSolutions.RationalPair.Heap
open FormalSW.PyAst
set_option Elab.async false

abbrev SetEntry := Pair × Nat
abbrev encodeEntry (entry : SetEntry) : HashKey × Mutable.Value := (key entry.1, .ref entry.2)
abbrev setObject (entries : List SetEntry) : Mutable.Object := .keySet (entries.map encodeEntry)
def EntryRows (heap : Mutable.Heap) (entries : List SetEntry) : Prop :=
  ∀ entry ∈ entries, heap[entry.2]? = some (object entry.1)

/-- Empty sets may still use scalar storage; tuple-key sets preserve their representatives. -/
structure SetAt (heap : Mutable.Heap) (address : Nat) (entries : List SetEntry) : Prop where
  stored : heap[address]? = some (setObject entries) ∨ (entries = [] ∧ heap[address]? = some (.set []))
  rows : EntryRows heap entries

def insertEntry (entries : List SetEntry) (entry : SetEntry) : List SetEntry :=
  if entries.any (fun old => decide (old.1 = entry.1)) then entries else entries ++ [entry]

theorem insertEntry_encoded (entries : List SetEntry) (entry : SetEntry) :
    Mutable.insertKey (entries.map encodeEntry) (encodeEntry entry) = (insertEntry entries entry).map encodeEntry := by
  simp only [Mutable.insertKey, Mutable.keyContains, List.any_map, Function.comp_def, encodeEntry, key_equal, insertEntry]
  split <;> simp

theorem mem_insertEntry (entries : List SetEntry) (entry : SetEntry) (pair : Pair) :
    pair ∈ (insertEntry entries entry).map Prod.fst ↔ pair ∈ entries.map Prod.fst ∨ pair = entry.1 := by
  unfold insertEntry
  split_ifs with duplicate
  · have present : entry.1 ∈ entries.map Prod.fst := by simpa using duplicate
    constructor
    · exact Or.inl
    · rintro (member | rfl)
      · exact member
      · exact present
  · simp

theorem entryRows_insert (heap : Mutable.Heap) (entries : List SetEntry) (entry : SetEntry)
    (rows : EntryRows heap entries) (row : heap[entry.2]? = some (object entry.1)) :
    EntryRows heap (insertEntry entries entry) := by
  unfold insertEntry
  split
  · exact rows
  · intro other member
    rcases List.mem_append.mp member with old | added
    · exact rows other old
    · simpa only [List.mem_singleton.mp added] using row

theorem SetAt.bound {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) : address < heap.length := by
  rcases found.stored with stored | ⟨_, stored⟩ <;> exact (List.getElem?_eq_some_iff.mp stored).1

theorem SetAt.row_write {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) (entry : SetEntry) (row : heap[entry.2]? = some (object entry.1))
    (replacement : List SetEntry) :
    (heap.set address (setObject replacement))[entry.2]? = some (object entry.1) := by
  have different : address ≠ entry.2 := by
    intro equal
    rcases found.stored with stored | ⟨_, stored⟩ <;>
      simp [equal, row, setObject, object] at stored
  simpa [List.getElem?_set, different, Ne.symm different] using row

theorem SetAt.write {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) (replacement : List SetEntry) (rows : EntryRows heap replacement) :
    SetAt (heap.set address (setObject replacement)) address replacement := by
  refine ⟨.inl ?_, ?_⟩
  · simp [found.bound]
  · intro entry member
    exact found.row_write entry (rows entry member) replacement

theorem SetAt.append {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) (extra : Mutable.Heap) : SetAt (heap ++ extra) address entries := by
  constructor
  · rcases found.stored with stored | ⟨empty, stored⟩
    · exact .inl (ProjectEulerSolutions.HeapCollection.prefix_found heap extra address _ stored)
    · exact .inr ⟨empty, ProjectEulerSolutions.HeapCollection.prefix_found heap extra address _ stored⟩
  · intro entry member
    exact ProjectEulerSolutions.HeapCollection.prefix_found heap extra entry.2 _ (found.rows entry member)

theorem SetAt.keyEntries {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) :
    (heap[address]?).bind Mutable.keyEntries = some (entries.map encodeEntry) := by
  rcases found.stored with stored | ⟨rfl, stored⟩ <;> simp [stored, Mutable.keyEntries, setObject]

theorem SetAt.read {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) :
    Mutable.setSourceItems (.ref address) heap = some (.ok (entries.map (fun entry => .ref entry.2)), heap) := by
  rcases found.stored with stored | ⟨rfl, stored⟩
  · simp [Mutable.setSourceItems, stored, setObject, List.map_map, Function.comp_def]
  · simp [Mutable.setSourceItems, stored]

/-- Insertion compares pair values and leaves the existing representative untouched on duplicates. -/
theorem SetAt.add {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) (entry : SetEntry) (row : heap[entry.2]? = some (object entry.1)) :
    Mutable.addItem address (.ref entry.2) heap =
      some (.ok (.scalar .none), heap.set address (setObject (insertEntry entries entry))) := by
  have hashed := (show Row heap (.ref entry.2) entry.1 from ⟨entry.2, rfl, row⟩).hash
  rcases found.stored with stored | ⟨rfl, stored⟩
  · simp [Mutable.addItem, stored, hashed, setObject, insertEntry_encoded, Except.map]
  · simp [Mutable.addItem, stored, hashed, setObject, insertEntry, Mutable.insertKey, Mutable.keyContains, Except.map]

end ProjectEulerSolutions.RationalPair.Heap
