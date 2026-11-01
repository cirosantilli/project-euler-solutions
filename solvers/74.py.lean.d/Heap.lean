import solvers.«74.py.lean.d».Positions

/-! Fresh path and position objects share a stable caller prefix throughout the forward walk. -/
namespace solvers.«74».heap
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P74Ast.module
abbrev number := factorials.number
abbrev numbers (path : List Nat) : List Mutable.Value := path.map number

def work (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage : Mutable.Heap) : Mutable.Heap :=
  base ++ (.list (numbers path) :: .dict (store.encode positions) :: garbage)

@[simp] theorem path_found (base : Mutable.Heap) (path : List Nat)
    (positions : store.Entries) (garbage : Mutable.Heap) :
    (work base path positions garbage)[base.length]? = some (.list (numbers path)) := by
  simp [work]

@[simp] theorem positions_found (base : Mutable.Heap) (path : List Nat)
    (positions : store.Entries) (garbage : Mutable.Heap) :
    store.Found (work base path positions garbage) (base.length + 1) positions := by
  simp [store.Found, work]

theorem old_found (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : base[address]? = some object) :
    (work base path positions garbage)[address]? = some object := by
  exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found

theorem work_append (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage extra : Mutable.Heap) :
    work base path positions garbage ++ extra = work base path positions (garbage ++ extra) := by
  simp [work, List.append_assoc]

theorem set_path (base : Mutable.Heap) (path replacement : List Nat)
    (positions : store.Entries) (garbage : Mutable.Heap) :
    (work base path positions garbage).set base.length (.list (numbers replacement)) =
      work base replacement positions garbage := by
  simp [work, List.set_append_right]

theorem set_positions (base : Mutable.Heap) (path : List Nat)
    (positions replacement : store.Entries) (garbage : Mutable.Heap) :
    (work base path positions garbage).set (base.length + 1) (.dict (store.encode replacement)) =
      work base path replacement garbage := by
  simp [work, List.set_append_right]

theorem set_old (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (inside : address < base.length) :
    (work base path positions garbage).set address object =
      work (base.set address object) path positions garbage := by
  simp only [work, List.set_append_left address object inside]

variable {globals : Mutable.Globals}

theorem path_length (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage : Mutable.Heap) (locals : HeapCalls.Env)
    (getPath : locals "path" = some (.ref base.length)) :
    Evaluates program locals (.builtinCall .len [.name "path"])
      (work base path positions garbage) (number path.length)
      (work base path positions garbage) globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup getPath) .nil)
  simp [primitive, Mutable.length, numbers, number, factorials.number]

theorem path_append (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage : Mutable.Heap) (locals : HeapCalls.Env) (current : Nat)
    (getPath : locals "path" = some (.ref base.length))
    (getCurrent : locals "current" = some (number current)) :
    Evaluates program locals (.call (.attribute (.name "path") "append") [.name "current"])
      (work base path positions garbage) (.scalar .none)
      (work base (path ++ [current]) positions garbage) globals := by
  have appended := Evaluates.append (globals := globals) (module := program)
    (Evaluates.lookup getPath) (path_found base path positions garbage)
    (Evaluates.lookup getCurrent) (path_found base path positions garbage)
  have values : numbers path ++ [number current] = numbers (path ++ [current]) := by
    simp only [numbers, List.map_append, List.map_cons, List.map_nil]
  rw [values, set_path] at appended
  exact appended

theorem write_position (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage : Mutable.Heap) (locals : HeapCalls.Env) (current : Nat)
    (getPath : locals "path" = some (.ref base.length))
    (getPositions : locals "positions" = some (.ref (base.length + 1)))
    (getCurrent : locals "current" = some (number current)) :
    BlockContinues program locals
      [.assign [.subscript (.name "positions") (.name "current")]
        (.builtinCall .len [.name "path"])] (work base path positions garbage) locals
      (work base path (store.insert positions current path.length) garbage) globals := by
  have written := store.insert_evaluates (globals := globals)
    (work base path positions garbage) locals
    "positions" "current" (base.length + 1) current path.length positions
    (.builtinCall .len [.name "path"]) (positions_found base path positions garbage)
    getPositions getCurrent (path_length base path positions garbage locals getPath)
  simpa only [set_positions] using written

theorem successor (base : Mutable.Heap) (path : List Nat) (positions : store.Entries)
    (garbage : Mutable.Heap) (locals : HeapCalls.Env) (current address : Nat)
    (getCurrent : locals "current" = some (number current))
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (found : base[address]? = some (.list factorials.values)) :
    ∃ extra, Evaluates program locals (.call (.name "next_term") [.name "current"])
      (work base path positions garbage)
      (number (ProjectEulerStatements.DigitFactorialChains.nextTerm current))
      (work base path positions (garbage ++ extra)) globals := by
  obtain ⟨extra, returned⟩ := next_term.result current address (work base path positions garbage)
    globals getTable (old_found base path positions garbage address _ found)
  refine ⟨extra, ?_⟩
  simpa only [work_append] using
    Evaluates.call (.cons (Evaluates.lookup getCurrent) .nil) returned

end solvers.«74».heap
