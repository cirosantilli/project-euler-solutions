import solvers.«62.source»
import ProjectEulerSolutions.Lib.CubicGroups
import Mathlib.Data.List.Nodup

/-! Live dictionary representation preserved by complete source grouping steps. -/
set_option Elab.async false
namespace solvers.«62».store
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.CubicPermutations ProjectEulerSolutions.CubicGroups
private abbrev program := Solvers.P62Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private def numbers (digits : List Nat) : List Mutable.Value := digits.map (fun n : Nat => number n)

structure Cell where
  group : Group
  keyAddress : Nat
  bucket : Nat

def Cell.entry (cell : Cell) : HashDict.Entry Mutable.Value :=
  (counterKey cell.group.digits, .ref cell.keyAddress, .ref cell.bucket)
def Cell.keyObject (cell : Cell) : Mutable.Object := .tuple (numbers cell.group.digits)
def Cell.bucketObject (cell : Cell) : Mutable.Object := .list [number cell.group.count, number cell.group.minimum]
def Cell.candidate (cell : Cell) : candidates.Row :=
  ⟨.ref cell.keyAddress, cell.bucket, cell.group.count, cell.group.minimum⟩
def Cell.update (cube : Nat) (cell : Cell) : Cell := { cell with group := ProjectEulerSolutions.CubicGroups.update cube cell.group }
def entries (cells : List Cell) : List (HashDict.Entry Mutable.Value) := cells.map Cell.entry
def logical (cells : List Cell) : List Group := cells.map Cell.group
def updated (cube : Nat) (cells : List Cell) : List Cell := cells.map (Cell.update cube)
def find (digits : List Nat) (cells : List Cell) : Option Cell :=
  cells.find? (fun cell => decide (cell.group.digits = digits))

structure Store (heap : Mutable.Heap) (dictionary : Nat) (cells : List Cell) : Prop where
  found : heap[dictionary]? = some (.keyDict (entries cells))
  keys : ∀ cell ∈ cells, heap[cell.keyAddress]? = some cell.keyObject
  values : ∀ cell ∈ cells, heap[cell.bucket]? = some cell.bucketObject
  signatures : (cells.map (fun cell => cell.group.digits)).Nodup
  buckets : (cells.map Cell.bucket).Nodup

def Store.append {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) (extra : Mutable.Heap) : Store (heap ++ extra) dictionary cells where
  found := ProjectEulerSolutions.HeapCollection.prefix_found heap extra dictionary _ view.found
  keys := fun cell member => ProjectEulerSolutions.HeapCollection.prefix_found heap extra cell.keyAddress _ (view.keys cell member)
  values := fun cell member => ProjectEulerSolutions.HeapCollection.prefix_found heap extra cell.bucket _ (view.values cell member)
  signatures := view.signatures
  buckets := view.buckets

@[simp] theorem update_entry (cube : Nat) (cell : Cell) : (cell.update cube).entry = cell.entry := by
  simp [Cell.entry, Cell.update]
@[simp] theorem update_signature (cube : Nat) (cell : Cell) :
    (cell.update cube).group.digits = cell.group.digits := update_digits cube cell.group
@[simp] theorem update_bucket (cube : Nat) (cell : Cell) : (cell.update cube).bucket = cell.bucket := rfl
@[simp] theorem update_keyAddress (cube : Nat) (cell : Cell) : (cell.update cube).keyAddress = cell.keyAddress := rfl
@[simp] theorem entries_updated (cube : Nat) (cells : List Cell) : entries (updated cube cells) = entries cells := by
  simp [entries, updated, List.map_map, Function.comp_def]
@[simp] theorem signatures_updated (cube : Nat) (cells : List Cell) :
    (updated cube cells).map (fun cell => cell.group.digits) = cells.map (fun cell => cell.group.digits) := by
  simp [updated, List.map_map, Function.comp_def]
@[simp] theorem buckets_updated (cube : Nat) (cells : List Cell) :
    (updated cube cells).map Cell.bucket = cells.map Cell.bucket := by
  simp [updated, List.map_map, Function.comp_def]

theorem lookup (digits : List Nat) (cells : List Cell) :
    HashDict.lookup (entries cells) (counterKey digits) = (find digits cells).map (fun cell => .ref cell.bucket) := by
  induction cells with
  | nil => rfl
  | cons cell rest ih =>
      by_cases same : cell.group.digits = digits
      · simp [entries, Cell.entry, HashDict.lookup_cons, counterKey_equal, find, List.find?, same]
      · simpa [entries, Cell.entry, HashDict.lookup_cons, counterKey_equal, find, List.find?, same] using ih

theorem find_member {digits : List Nat} {cells : List Cell} {cell : Cell}
    (selected : find digits cells = some cell) : cell ∈ cells ∧ cell.group.digits = digits :=
  ⟨List.mem_of_find?_eq_some selected,
    of_decide_eq_true (List.find?_some (p := fun other : Cell => decide (other.group.digits = digits)) selected)⟩

theorem find_none {digits : List Nat} {cells : List Cell} (absent : find digits cells = none) :
    ∀ cell ∈ cells, cell.group.digits ≠ digits := by
  simpa [find] using List.find?_eq_none.mp absent

theorem Store.signature_injective {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) {left right : Cell} (leftMem : left ∈ cells) (rightMem : right ∈ cells)
    (same : left.group.digits = right.group.digits) : left = right :=
  List.inj_on_of_nodup_map view.signatures leftMem rightMem same

theorem Store.bucket_injective {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) {left right : Cell} (leftMem : left ∈ cells) (rightMem : right ∈ cells)
    (same : left.bucket = right.bucket) : left = right :=
  List.inj_on_of_nodup_map view.buckets leftMem rightMem same

theorem logical_updated (cube : Nat) (cells : List Cell) :
    logical (updated cube cells) = (logical cells).map (update cube) := by
  simp [logical, updated, Cell.update, List.map_map, Function.comp_def]

theorem contains_find (cube : Nat) (cells : List Cell) :
    contains cube (logical cells) = (find (signature cube) cells).isSome := by
  induction cells with
  | nil => rfl
  | cons cell rest ih =>
      by_cases same : cell.group.digits = signature cube
      · simp [contains, logical, find, List.find?, same]
      · simpa [contains, logical, find, List.find?, same] using ih

theorem Store.dictionary_key_ne {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) (cell : Cell) (member : cell ∈ cells) : dictionary ≠ cell.keyAddress := by
  intro same
  have found := view.keys cell member
  simp [← same, view.found, Cell.keyObject] at found

theorem Store.dictionary_bucket_ne {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) (cell : Cell) (member : cell ∈ cells) : dictionary ≠ cell.bucket := by
  intro same
  have found := view.values cell member
  simp [← same, view.found, Cell.bucketObject] at found

theorem Store.bucket_key_ne {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) (bucketCell keyCell : Cell)
    (bucketMember : bucketCell ∈ cells) (keyMember : keyCell ∈ cells) : bucketCell.bucket ≠ keyCell.keyAddress := by
  intro same
  have found := view.values bucketCell bucketMember
  simp [same, view.keys keyCell keyMember, Cell.bucketObject, Cell.keyObject] at found

@[simp] theorem update_keyObject (cube : Nat) (cell : Cell) : (cell.update cube).keyObject = cell.keyObject := by
  simp [Cell.keyObject]

/-- Updating a selected bucket preserves every retained key and every other bucket. -/
theorem Store.hit {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) (cube : Nat) (chosen : Cell) (member : chosen ∈ cells)
    (same : chosen.group.digits = signature cube) :
    Store (heap.set chosen.bucket (chosen.update cube).bucketObject) dictionary (updated cube cells) where
  found := by
    simpa only [entries_updated, List.getElem?_set_ne (view.dictionary_bucket_ne chosen member).symm] using view.found
  keys := by
    intro cell cellMember
    obtain ⟨old, oldMember, rfl⟩ := List.mem_map.mp cellMember
    simpa only [update_keyAddress, update_keyObject,
      List.getElem?_set_ne (view.bucket_key_ne chosen old member oldMember)] using view.keys old oldMember
  values := by
    intro cell cellMember
    obtain ⟨old, oldMember, rfl⟩ := List.mem_map.mp cellMember
    by_cases matching : old.group.digits = signature cube
    · have equal := view.signature_injective oldMember member (matching.trans same.symm)
      subst old
      exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp (view.values chosen member)).1
    · have different : chosen.bucket ≠ old.bucket := by
        intro equal
        have equalCells := view.bucket_injective member oldMember equal
        subst old
        exact matching same
      simpa only [update_bucket, Cell.update, ProjectEulerSolutions.CubicGroups.update,
        if_neg matching, List.getElem?_set_ne different] using view.values old oldMember
  signatures := by simpa only [signatures_updated] using view.signatures
  buckets := by simpa only [buckets_updated] using view.buckets

theorem insert_absent (items : List (HashDict.Entry Mutable.Value)) (query : HashKey)
    (original value : Mutable.Value) (absent : HashDict.lookup items query = none) :
    HashDict.insert items query original value = items ++ [(query, original, value)] := by
  induction items with
  | nil => rfl
  | cons item rest ih =>
      cases same : HashKey.equal item.1 query
      · have missing : HashDict.lookup rest query = none := by simpa [HashDict.lookup_cons, same] using absent
        simp [HashDict.insert, same, ih missing]
      · simp [HashDict.lookup_cons, same] at absent

def hitHeap (cube : Nat) (heap : Mutable.Heap) (chosen : Cell) : Mutable.Heap :=
  (heap ++ groups.signatureObjects cube).set chosen.bucket (chosen.update cube).bucketObject

/-- The source hit branch realizes the logical update while retaining the complete live table. -/
theorem grouping_hit (cube : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (cells : List Cell) (view : Store heap dictionary cells) (chosen : Cell)
    (selected : find (signature cube) cells = some chosen)
    (getGroups : env "groups" = some (.ref dictionary)) (getCube : env "cube" = some (number cube)) :
    BlockContinues program env groups.groupingCode heap
      (setLocal env "sig" (.ref (heap.length + 2))) (hitHeap cube heap chosen) ∧
      Store (hitHeap cube heap chosen) dictionary (updated cube cells) ∧
      logical (updated cube cells) = record cube (logical cells) := by
  obtain ⟨member, same⟩ := find_member selected
  have looked : HashDict.lookup (entries cells) (key cube) = some (.ref chosen.bucket) := by
    rw [key, lookup, selected]
    rfl
  have executed := groups.grouping_hit cube chosen.group.count chosen.group.minimum heap env dictionary chosen.bucket
    (entries cells) getGroups getCube view.found looked (view.values chosen member)
  have retained := (view.append (groups.signatureObjects cube)).hit cube chosen member same
  have bucket : (chosen.update cube).bucketObject =
      .list [number (chosen.group.count + 1), number (min chosen.group.minimum cube)] := by
    simp [Cell.bucketObject, Cell.update, ProjectEulerSolutions.CubicGroups.update, bump, same]
  refine ⟨?_, retained, ?_⟩
  · simpa only [hitHeap, bucket] using executed
  · rw [record, contains_find, selected]
    exact logical_updated cube cells

def Cell.fresh (cube size : Nat) : Cell := ⟨ProjectEulerSolutions.CubicGroups.fresh cube, size + 2, size + 3⟩
def allocated (cube : Nat) (heap : Mutable.Heap) : Mutable.Heap :=
  heap ++ groups.signatureObjects cube ++ [(Cell.fresh cube heap.length).bucketObject]
def insertedHeap (cube : Nat) (heap : Mutable.Heap) (dictionary : Nat) (cells : List Cell) : Mutable.Heap :=
  (allocated cube heap).set dictionary (.keyDict (entries (cells ++ [Cell.fresh cube heap.length])))

@[simp] theorem allocated_length (cube : Nat) (heap : Mutable.Heap) : (allocated cube heap).length = heap.length + 4 := by
  simp [allocated, groups.signatureObjects]

theorem allocated_old (cube : Nat) (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (allocated cube heap)[address]? = some object := by
  simpa only [allocated, List.append_assoc] using ProjectEulerSolutions.HeapCollection.prefix_found heap
    (groups.signatureObjects cube ++ [(Cell.fresh cube heap.length).bucketObject]) address object found

theorem allocated_key (cube : Nat) (heap : Mutable.Heap) :
    (allocated cube heap)[heap.length + 2]? = some (Cell.fresh cube heap.length).keyObject := by
  simp [allocated, groups.signatureObjects, Cell.fresh, Cell.keyObject, ProjectEulerSolutions.CubicGroups.fresh, numbers]
  rfl

theorem allocated_bucket (cube : Nat) (heap : Mutable.Heap) :
    (allocated cube heap)[heap.length + 3]? = some (Cell.fresh cube heap.length).bucketObject := by
  simp [allocated, groups.signatureObjects]

/-- A fresh group is appended in insertion order without moving existing representatives. -/
theorem Store.miss {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) (cube : Nat) (absent : find (signature cube) cells = none) :
    Store (insertedHeap cube heap dictionary cells) dictionary (cells ++ [Cell.fresh cube heap.length]) := by
  have dictionaryBound := (List.getElem?_eq_some_iff.mp view.found).1
  have keyNe : dictionary ≠ heap.length + 2 := by omega
  have bucketNe : dictionary ≠ heap.length + 3 := by omega
  have noSignature : signature cube ∉ cells.map (fun cell => cell.group.digits) := by
    intro member
    obtain ⟨cell, cellMember, same⟩ := List.mem_map.mp member
    exact find_none absent cell cellMember same
  have noBucket : heap.length + 3 ∉ cells.map Cell.bucket := by
    intro member
    obtain ⟨cell, cellMember, same⟩ := List.mem_map.mp member
    have bound := (List.getElem?_eq_some_iff.mp (view.values cell cellMember)).1
    omega
  constructor
  · exact List.getElem?_set_self (by simp only [allocated_length]; omega)
  · intro cell member
    rcases List.mem_append.mp member with old | fresh
    · rw [insertedHeap, List.getElem?_set_ne (view.dictionary_key_ne cell old)]
      exact allocated_old cube heap cell.keyAddress cell.keyObject (view.keys cell old)
    · have equal : cell = Cell.fresh cube heap.length := by simpa using fresh
      subst cell
      change (insertedHeap cube heap dictionary cells)[heap.length + 2]? = _
      rw [insertedHeap, List.getElem?_set_ne keyNe]
      exact allocated_key cube heap
  · intro cell member
    rcases List.mem_append.mp member with old | fresh
    · rw [insertedHeap, List.getElem?_set_ne (view.dictionary_bucket_ne cell old)]
      exact allocated_old cube heap cell.bucket cell.bucketObject (view.values cell old)
    · have equal : cell = Cell.fresh cube heap.length := by simpa using fresh
      subst cell
      change (insertedHeap cube heap dictionary cells)[heap.length + 3]? = _
      rw [insertedHeap, List.getElem?_set_ne bucketNe]
      exact allocated_bucket cube heap
  · simpa [Cell.fresh, ProjectEulerSolutions.CubicGroups.fresh, List.nodup_append] using And.intro view.signatures noSignature
  · simpa [Cell.fresh, List.nodup_append] using And.intro view.buckets noBucket

theorem grouping_miss (cube : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (cells : List Cell) (view : Store heap dictionary cells) (absent : find (signature cube) cells = none)
    (getGroups : env "groups" = some (.ref dictionary)) (getCube : env "cube" = some (number cube)) :
    BlockContinues program env groups.groupingCode heap (setLocal env "sig" (.ref (heap.length + 2)))
      (insertedHeap cube heap dictionary cells) ∧
      Store (insertedHeap cube heap dictionary cells) dictionary (cells ++ [Cell.fresh cube heap.length]) ∧
      logical (cells ++ [Cell.fresh cube heap.length]) = record cube (logical cells) := by
  have missing : HashDict.lookup (entries cells) (key cube) = none := by rw [key, lookup, absent]; rfl
  have executed := groups.grouping_miss cube heap env dictionary (entries cells) getGroups getCube view.found missing
  have inserted := insert_absent (entries cells) (key cube) (.ref (heap.length + 2))
    (.ref (heap ++ groups.signatureObjects cube).length) missing
  rw [inserted] at executed
  refine ⟨?_, view.miss cube absent, ?_⟩
  · simpa [insertedHeap, allocated, entries, Cell.entry, Cell.fresh, ProjectEulerSolutions.CubicGroups.fresh,
      Cell.bucketObject, groups.signatureObjects, key] using executed
  · rw [record, contains_find, absent]
    simp [logical, Cell.fresh]

/-- A complete source grouping step preserves the live representation and realizes the ordered logical record. -/
theorem grouping_result (cube : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (cells : List Cell) (view : Store heap dictionary cells)
    (getGroups : env "groups" = some (.ref dictionary)) (getCube : env "cube" = some (number cube)) :
    ∃ finalHeap finalCells, BlockContinues program env groups.groupingCode heap
      (setLocal env "sig" (.ref (heap.length + 2))) finalHeap ∧ Store finalHeap dictionary finalCells ∧
      logical finalCells = record cube (logical cells) := by
  cases selected : find (signature cube) cells with
  | none =>
      obtain ⟨executed, retained, recorded⟩ := grouping_miss cube heap env dictionary cells view selected getGroups getCube
      exact ⟨_, _, executed, retained, recorded⟩
  | some chosen =>
      obtain ⟨executed, retained, recorded⟩ := grouping_hit cube heap env dictionary cells view chosen selected getGroups getCube
      exact ⟨_, _, executed, retained, recorded⟩

def Above (floor : Nat) (cells : List Cell) : Prop :=
  ∀ cell ∈ cells, floor ≤ cell.keyAddress ∧ floor ≤ cell.bucket

theorem Above.updated {floor : Nat} {cells : List Cell} (above : Above floor cells) (cube : Nat) :
    Above floor (updated cube cells) := by
  intro cell member
  obtain ⟨old, oldMember, rfl⟩ := List.mem_map.mp member
  exact above old oldMember

theorem Above.fresh {floor : Nat} {cells : List Cell} (above : Above floor cells) (cube size : Nat)
    (bounded : floor ≤ size) : Above floor (cells ++ [Cell.fresh cube size]) := by
  intro cell member
  rcases List.mem_append.mp member with old | fresh
  · exact above cell old
  · have equal : cell = Cell.fresh cube size := by simpa using fresh
    subst cell
    exact ⟨by change floor ≤ size + 2; omega, by change floor ≤ size + 3; omega⟩

theorem hitHeap_length (cube : Nat) (heap : Mutable.Heap) (chosen : Cell) :
    (hitHeap cube heap chosen).length = heap.length + 3 := by simp [hitHeap, groups.signatureObjects]

theorem insertedHeap_length (cube : Nat) (heap : Mutable.Heap) (dictionary : Nat) (cells : List Cell) :
    (insertedHeap cube heap dictionary cells).length = heap.length + 4 := by simp [insertedHeap]

theorem hitHeap_prefix (cube : Nat) (heap : Mutable.Heap) (chosen : Cell) (floor : Nat)
    (bounded : floor ≤ heap.length) (fresh : floor ≤ chosen.bucket) :
    (hitHeap cube heap chosen).take floor = heap.take floor := by
  rw [hitHeap, List.take_set_of_le fresh]
  exact List.take_append_of_le_length bounded

theorem insertedHeap_prefix (cube : Nat) (heap : Mutable.Heap) (dictionary : Nat) (cells : List Cell)
    (floor : Nat) (bounded : floor ≤ heap.length) (fresh : floor ≤ dictionary) :
    (insertedHeap cube heap dictionary cells).take floor = heap.take floor := by
  rw [insertedHeap, List.take_set_of_le fresh]
  simpa only [allocated, List.append_assoc] using
    (List.take_append_of_le_length (l₂ := groups.signatureObjects cube ++ [(Cell.fresh cube heap.length).bucketObject]) bounded)

/-- The loop-ready grouping rule also preserves caller objects and fresh-address bounds. -/
theorem grouping_preserves (cube : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (cells : List Cell) (view : Store heap dictionary cells)
    (getGroups : env "groups" = some (.ref dictionary)) (getCube : env "cube" = some (number cube))
    (floor : Nat) (freshDictionary : floor ≤ dictionary) (above : Above floor cells) :
    ∃ finalHeap finalCells, BlockContinues program env groups.groupingCode heap
      (setLocal env "sig" (.ref (heap.length + 2))) finalHeap ∧ Store finalHeap dictionary finalCells ∧
      logical finalCells = record cube (logical cells) ∧ Above floor finalCells ∧
      finalHeap.take floor = heap.take floor ∧ heap.length ≤ finalHeap.length := by
  have bound : floor ≤ heap.length := by
    have inside := (List.getElem?_eq_some_iff.mp view.found).1
    omega
  cases selected : find (signature cube) cells with
  | none =>
      obtain ⟨executed, retained, recorded⟩ := grouping_miss cube heap env dictionary cells view selected getGroups getCube
      exact ⟨_, _, executed, retained, recorded, above.fresh cube heap.length bound,
        insertedHeap_prefix cube heap dictionary cells floor bound freshDictionary, by rw [insertedHeap_length]; omega⟩
  | some chosen =>
      obtain ⟨executed, retained, recorded⟩ := grouping_hit cube heap env dictionary cells view chosen selected getGroups getCube
      have member := (find_member selected).1
      exact ⟨_, _, executed, retained, recorded, above.updated cube,
        hitHeap_prefix cube heap chosen floor bound (above chosen member).2, by rw [hitHeap_length]; omega⟩

/-- The first insertion promotes the real empty dictionary and establishes the invariant. -/
theorem initial_store (cube : Nat) (heap : Mutable.Heap) (dictionary : Nat)
    (found : heap[dictionary]? = some (.dict [])) :
    Store (insertedHeap cube heap dictionary []) dictionary [Cell.fresh cube heap.length] := by
  have bound := (List.getElem?_eq_some_iff.mp found).1
  have keyNe : dictionary ≠ heap.length + 2 := by omega
  have bucketNe : dictionary ≠ heap.length + 3 := by omega
  constructor
  · exact List.getElem?_set_self (by simp only [allocated_length]; omega)
  · intro cell member
    have equal : cell = Cell.fresh cube heap.length := by simpa using member
    subst cell
    change (insertedHeap cube heap dictionary [])[heap.length + 2]? = _
    rw [insertedHeap, List.getElem?_set_ne keyNe]
    exact allocated_key cube heap
  · intro cell member
    have equal : cell = Cell.fresh cube heap.length := by simpa using member
    subst cell
    change (insertedHeap cube heap dictionary [])[heap.length + 3]? = _
    rw [insertedHeap, List.getElem?_set_ne bucketNe]
    exact allocated_bucket cube heap
  · simp
  · simp

theorem grouping_initial (cube : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (getGroups : env "groups" = some (.ref dictionary)) (getCube : env "cube" = some (number cube))
    (found : heap[dictionary]? = some (.dict [])) :
    BlockContinues program env groups.groupingCode heap (setLocal env "sig" (.ref (heap.length + 2)))
      (insertedHeap cube heap dictionary []) ∧
      Store (insertedHeap cube heap dictionary []) dictionary [Cell.fresh cube heap.length] := by
  refine ⟨?_, initial_store cube heap dictionary found⟩
  simpa [insertedHeap, allocated, entries, Cell.entry, Cell.fresh, ProjectEulerSolutions.CubicGroups.fresh,
    Cell.bucketObject, groups.signatureObjects, key] using
    groups.grouping_initial cube heap env dictionary getGroups getCube found

def Store.candidates {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) : candidates.View heap dictionary (cells.map Cell.candidate) where
  object := .keyDict (entries cells)
  found := view.found
  fields := by simp [dictionaryItemFields, entries, Cell.entry, Cell.candidate, candidates.Row.fields,
    List.map_map, Function.comp_def]
  buckets := by
    intro row member
    obtain ⟨cell, cellMember, rfl⟩ := List.mem_map.mp member
    exact view.values cell cellMember

theorem matching_candidates (target : Int) (cells : List Cell) :
    candidates.selected target (cells.map Cell.candidate) = matching target (logical cells) := by
  simp [candidates.selected, matching, logical, Cell.candidate, List.filterMap_map]

def resetHeap (heap : Mutable.Heap) (dictionary : Nat) (garbage : Mutable.Heap) : Mutable.Heap :=
  (ProjectEulerSolutions.HeapCollection.working heap [] garbage).set dictionary (.keyDict [])

theorem Store.reset {heap : Mutable.Heap} {dictionary : Nat} {cells : List Cell}
    (view : Store heap dictionary cells) (garbage : Mutable.Heap) : Store (resetHeap heap dictionary garbage) dictionary [] := by
  constructor
  · apply List.getElem?_set_self
    have bound := (List.getElem?_eq_some_iff.mp view.found).1
    simp only [ProjectEulerSolutions.HeapCollection.working, List.length_append, List.length_cons]
    omega
  · simp
  · simp
  · simp
  · simp

theorem resetHeap_prefix (heap : Mutable.Heap) (dictionary : Nat) (garbage : Mutable.Heap) (floor : Nat)
    (bounded : floor ≤ heap.length) (fresh : floor ≤ dictionary) :
    (resetHeap heap dictionary garbage).take floor = heap.take floor := by
  rw [resetHeap, List.take_set_of_le fresh]
  exact List.take_append_of_le_length bounded

theorem resetHeap_length (heap : Mutable.Heap) (dictionary : Nat) (garbage : Mutable.Heap) :
    (resetHeap heap dictionary garbage).length = heap.length + 1 + garbage.length := by
  simp only [resetHeap, ProjectEulerSolutions.HeapCollection.working, List.length_set, List.length_append, List.length_cons]
  omega

/-- Selection reads the same logical table that the grouping rule maintains. -/
theorem width_returns (target : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (cells : List Cell) (view : Store heap dictionary cells)
    (getGroups : env "groups" = some (.ref dictionary)) (getK : env "k" = some (number target))
    (first : Int) (rest : List Int) (selected : matching target (logical cells) = first :: rest) :
    ∃ garbage, BlockReturns program env groups.widthBody heap (number (rest.foldl min first))
      (ProjectEulerSolutions.HeapCollection.working heap ((matching target (logical cells)).map number) garbage) := by
  obtain ⟨garbage, returned⟩ := candidates.width_returns target heap env dictionary (cells.map Cell.candidate)
    view.candidates getGroups getK first rest (by simpa only [matching_candidates] using selected)
  exact ⟨garbage, by simpa only [matching_candidates] using returned⟩

/-- An unsuccessful width transition restores the empty invariant without changing caller objects. -/
theorem width_continues (target length : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (cells : List Cell) (view : Store heap dictionary cells)
    (getGroups : env "groups" = some (.ref dictionary)) (getK : env "k" = some (number target))
    (getLength : env "length" = some (number length)) (empty : matching target (logical cells) = [])
    (floor : Nat) (freshDictionary : floor ≤ dictionary) :
    ∃ garbage, BlockContinues program env groups.widthBody heap
      (setLocal (setLocal env "candidates" (.ref heap.length)) "current_len" (number length))
      (resetHeap heap dictionary garbage) ∧ Store (resetHeap heap dictionary garbage) dictionary [] ∧
      (resetHeap heap dictionary garbage).take floor = heap.take floor ∧
      heap.length ≤ (resetHeap heap dictionary garbage).length := by
  obtain ⟨garbage, executed⟩ := candidates.width_continues target length heap env dictionary (entries cells)
    (cells.map Cell.candidate) view.candidates view.found getGroups getK getLength
    (by simpa only [matching_candidates] using empty)
  refine ⟨garbage, executed, view.reset garbage, ?_, ?_⟩
  · apply resetHeap_prefix heap dictionary garbage floor _ freshDictionary
    have bound := (List.getElem?_eq_some_iff.mp view.found).1
    omega
  · rw [resetHeap_length]
    omega

end solvers.«62».store

namespace solvers.«62»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

theorem grouping_store_source_result (cube : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (dictionary : Nat) (cells : List store.Cell) (view : store.Store heap dictionary cells) (floor : Nat)
    (getGroups : env "groups" = some (.ref dictionary)) (getCube : env "cube" = some (.scalar (.int cube)))
    (freshDictionary : floor ≤ dictionary) (above : store.Above floor cells) :
    ∃ finalHeap finalCells, BlockContinues Solvers.P62Ast.module env groups.groupingCode heap
      (setLocal env "sig" (.ref (heap.length + 2))) finalHeap ∧ store.Store finalHeap dictionary finalCells ∧
      store.logical finalCells = ProjectEulerSolutions.CubicGroups.record cube (store.logical cells) ∧
      store.Above floor finalCells ∧ finalHeap.take floor = heap.take floor ∧ heap.length ≤ finalHeap.length :=
  store.grouping_preserves cube heap env dictionary cells view getGroups getCube floor freshDictionary above

end solvers.«62»
