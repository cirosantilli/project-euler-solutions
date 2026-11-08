import solvers.«84.py.lean.d».TransitionBindings
import lang.py.FormalSW.DictionaryItemsCalls

namespace solvers.«84».transition_row.items
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary resolve_landing
set_option Elab.async false

def body : List Stmt := bindings.body ++ landingBody
def receiver : Expr := .subscript (.name "landings")
  (.binOp (.binOp (.name "pos") .add (.name "move")) .mod (.constant (.int 40)))

theorem source : landingCode = [.forDictionaryItems bindings.target receiver landingBody []] := rfl

def fields (entries : List weights.Entry) : List (Mutable.Value × Mutable.Value) :=
  entries.map fun entry => (.ref entry.address, .scalar (.int entry.weight))
def keys (entries : List weights.Entry) : List SequenceSource := entries.map fun entry => .ref entry.address
def cursor (address : Nat) (entries : List weights.Entry) (position : Nat) : Stmt :=
  .dictionaryItemsStepLoop "$formalsw.unpack" address (keys entries) position body []

theorem represented (entries : List weights.Entry) :
    dictionaryItemFields (.keyDict (weights.encode entries)) = some (fields entries) := by
  simp [dictionaryItemFields, weights.encode, fields, List.map_map, Function.comp_def]

theorem next (heap : Mutable.Heap) (address position : Nat) (entries : List weights.Entry)
    (found : heap[address]? = some (.keyDict (weights.encode entries))) :
    dictionaryItemsNext address (keys entries) position heap =
      some (.ok (entries[position]?.map fun entry => (.ref entry.address, .scalar (.int entry.weight))), heap) := by
  simpa [keys, fields, List.map_map, Function.comp_def, sequenceSource, List.getElem?_map] using
    dictionaryItemsNext_fields (position := position) found (represented entries)

def applyEntry (doubles count : Nat) (isDouble : Bool) (entries : Entries) (entry : weights.Entry) : Entries :=
  add entries (Monopoly.encode entry.landing.1.toNat (nextDoubles entry.landing.2 isDouble doubles)) (entry.weight.toNat * count)

def pairHeap (heap : Mutable.Heap) (entry : weights.Entry) : Mutable.Heap :=
  heap ++ [.tuple [.ref entry.address, .scalar (.int entry.weight)]]

theorem allocated (heap : Mutable.Heap) (entry : weights.Entry) :
    Mutable.allocateTuple [.ref entry.address, .scalar (.int entry.weight)] heap =
      some (.ok (.ref heap.length), pairHeap heap entry) := by
  simp [Mutable.allocateTuple, pairHeap, Except.map]

def HeapFrame (before after : Mutable.Heap) (root : Nat) : Prop :=
  ∀ (address : Nat) (object : Mutable.Object), address ≠ root → before[address]? = some object → after[address]? = some object

theorem heap_frame (heap extra : Mutable.Heap) (root : Nat) (replacement : Entries) :
    HeapFrame heap ((heap ++ extra).set root (.dict (encode replacement))) root := by
  intro address object different found
  exact (List.getElem?_set_ne different.symm).trans (HeapCollection.prefix_found _ _ _ _ found)

theorem HeapFrame.tuples {before after : Mutable.Heap} {root : Nat} (frame : HeapFrame before after root)
    (entries : Entries) (found : Found before root entries) :
    ∀ (address : Nat) (values : List Mutable.Value), before[address]? = some (.tuple values) → after[address]? = some (.tuple values) := by
  intro address values present
  apply frame address _ _ present
  intro same
  subst address
  change before[root]? = some (.dict (encode entries)) at found
  rw [present] at found
  cases found

theorem keys_frame {before after : Mutable.Heap} {root : Nat} (frame : HeapFrame before after root)
    (entries : Entries) (found : Found before root entries) (original : List weights.Entry) (valid : weights.Valid before original) :
    weights.Valid after original := by
  intro entry member
  exact TuplePool.View.frame before after (frame.tuples entries found) _ _ (valid entry member)

theorem key_tuple (heap : Mutable.Heap) (entry : weights.Entry)
    (key : TuplePool.View heap (.tuple (pool.pair entry.landing.1 entry.landing.2)) (.ref entry.address)) :
    heap[entry.address]? = some (.tuple [.scalar (.int entry.landing.1), .scalar (.bool entry.landing.2)]) := by
  have view := (TuplePool.view_scalars heap [.int entry.landing.1, .bool entry.landing.2] (.ref entry.address)).mp key
  obtain ⟨address, equal, present⟩ := view
  cases equal
  exact present

def rebound : List String := "$formalsw.unpack" :: "next_doubles" :: "target" :: bindings.rebound
def LocalsFrame (before after : HeapCalls.Env) : Prop := ∀ name, name ∉ rebound → after name = before name

theorem locals_frame (env : HeapCalls.Env) (pairAddress doubles : Nat) (isDouble : Bool) (entry : weights.Entry) :
    LocalsFrame env (landingLocals (bindings.locals (setLocal env "$formalsw.unpack" (.ref pairAddress)) entry)
      entry.landing.1.toNat doubles entry.landing.2 isDouble) := by
  intro name keep
  simp only [rebound, List.mem_cons, not_or] at keep
  simp only [landingLocals, setLocal, if_neg keep.2.1, if_neg keep.2.2.1]
  rw [bindings.unchanged _ _ name keep.2.2.2]
  simp [setLocal, keep.1]

end solvers.«84».transition_row.items
