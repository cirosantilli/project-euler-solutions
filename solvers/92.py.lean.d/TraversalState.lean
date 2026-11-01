import solvers.«92.py.lean.d».Bindings
import lang.py.FormalSW.DictionaryItemsCalls

namespace solvers.«92».traversal
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := rows.program
abbrev body : List Stmt := bindings.body ++ [rows.loop]
abbrev loop : Stmt := .forDictionaryItems bindings.target (.name "counts") [rows.loop] []

def fields (entries : table.Entries) : List (Mutable.Value × Mutable.Value) :=
  entries.map fun entry => (.ref entry.2.1, table.number entry.2.2)

def keys (entries : table.Entries) : List SequenceSource :=
  entries.map fun entry => .ref entry.2.1

def cursor (address : Nat) (entries : table.Entries) (position : Nat) : Stmt :=
  .dictionaryItemsStepLoop "$formalsw.unpack" address (keys entries) position body []

theorem source :
    (match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[2]? with
      | some (Stmt.forIn _ _ prefixBody _) => prefixBody[2]?
      | _ => none) = some loop := rfl

theorem fields_found (heap : Mutable.Heap) (address : Nat) (entries : table.Entries)
    (found : table.Found heap address entries) :
    ∃ object, heap[address]? = some object ∧ dictionaryItemFields object = some (fields entries) := by
  rcases found with found | ⟨rfl, found⟩
  · refine ⟨.keyDict (table.encode entries), found, ?_⟩
    simp [dictionaryItemFields, table.encode, fields, List.map_map, Function.comp_def]
  · exact ⟨.dict [], found, rfl⟩

theorem next (heap : Mutable.Heap) (address position : Nat) (entries : table.Entries)
    (found : table.Found heap address entries) :
    dictionaryItemsNext address (keys entries) position heap =
      some (.ok (entries[position]?.map fun entry => (.ref entry.2.1, table.number entry.2.2)), heap) := by
  obtain ⟨object, present, represented⟩ := fields_found heap address entries found
  simpa [keys, fields, List.map_map, Function.comp_def, sequenceSource, List.getElem?_map] using
    dictionaryItemsNext_fields (position := position) present represented

theorem found_frame (before after : Mutable.Heap) (source destination : Nat)
    (entries : table.Entries) (found : table.Found before source entries)
    (different : source ≠ destination) (frame : rows.HeapFrame before after destination) :
    table.Found after source entries := by
  rcases found with found | ⟨rfl, found⟩
  · exact Or.inl (frame _ _ different found)
  · exact Or.inr ⟨rfl, frame _ _ different found⟩

theorem keys_frame (before after : Mutable.Heap) (destination : Nat)
    (entries original : table.Entries) (valid : table.Keys before entries)
    (found : table.Found before destination original) (frame : rows.HeapFrame before after destination) :
    table.Keys after entries := by
  intro entry member
  have present := valid entry member
  have different : entry.2.1 ≠ destination := by
    intro same
    rw [same] at present
    rcases found with found | ⟨_, found⟩ <;>
      rw [present] at found <;> simp [table.tuple] at found
  exact frame _ _ different present

def rebound : List String := "$formalsw.unpack" :: "digit" :: "key" :: bindings.rebound

def LocalsFrame (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ rebound → after name = before name

theorem locals_frame (before after : HeapCalls.Env) (pairAddress keyAddress ways : Nat) (state : Prefix.State)
    (frame : rows.LocalsFrame
      (bindings.locals (setLocal before "$formalsw.unpack" (.ref pairAddress)) keyAddress ways state) after) :
    LocalsFrame before after := by
  intro name keep
  simp only [rebound, List.mem_cons, not_or] at keep
  rw [frame name keep.2.1 keep.2.2.1,
    bindings.unchanged _ _ _ _ name keep.2.2.2]
  simp [setLocal, keep.1]

theorem local_following (before after : HeapCalls.Env) (frame : LocalsFrame before after) :
    after "following" = before "following" := frame _ (by decide)

theorem local_ceiling (before after : HeapCalls.Env) (frame : LocalsFrame before after) :
    after "ceiling" = before "ceiling" := frame _ (by decide)

end solvers.«92».traversal
