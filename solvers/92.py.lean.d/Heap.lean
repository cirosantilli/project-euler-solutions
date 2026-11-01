import solvers.«92.py.lean.d».Store
import lang.py.FormalSW.MembershipCalls

namespace solvers.«92».heap
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := store.number
abbrev numbers (path : List Nat) := path.map number
abbrev constants (path : List Nat) : List Constant := path.map (fun n : Nat => .int n)

def work (base : Mutable.Heap) (path positions : List Nat) (garbage : Mutable.Heap) : Mutable.Heap :=
  base ++ (.list (numbers path) :: .set (constants positions) :: garbage)

@[simp] theorem path_found (base : Mutable.Heap) (path positions : List Nat) (garbage : Mutable.Heap) :
    (work base path positions garbage)[base.length]? = some (.list (numbers path)) := by simp [work]

@[simp] theorem positions_found (base : Mutable.Heap) (path positions : List Nat) (garbage : Mutable.Heap) :
    (work base path positions garbage)[base.length + 1]? = some (.set (constants positions)) := by simp [work]

theorem old_found (base : Mutable.Heap) (path positions : List Nat) (garbage : Mutable.Heap)
    (address : Nat) (object : Mutable.Object) (found : base[address]? = some object) :
    (work base path positions garbage)[address]? = some object :=
  (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found

theorem work_append (base : Mutable.Heap) (path positions : List Nat) (garbage extra : Mutable.Heap) :
    work base path positions garbage ++ extra = work base path positions (garbage ++ extra) := by
  simp [work, List.append_assoc]

theorem set_path (base : Mutable.Heap) (path replacement positions : List Nat) (garbage : Mutable.Heap) :
    (work base path positions garbage).set base.length (.list (numbers replacement)) =
      work base replacement positions garbage := by simp [work, List.set_append_right]

theorem set_positions (base : Mutable.Heap) (path positions replacement : List Nat) (garbage : Mutable.Heap) :
    (work base path positions garbage).set (base.length + 1) (.set (constants replacement)) =
      work base path replacement garbage := by simp [work, List.set_append_right]

theorem set_old (base : Mutable.Heap) (path positions : List Nat) (garbage : Mutable.Heap)
    (address : Nat) (object : Mutable.Object) (inside : address < base.length) :
    (work base path positions garbage).set address object = work (base.set address object) path positions garbage := by
  simp only [work, List.set_append_left address object inside]

theorem contains (positions : List Nat) (current : Nat) :
    ScalarSet.contains (constants positions) (.int current) = decide (current ∈ positions) := by
  have tested := ScalarSet.contains_ints (positions.map (fun n : Nat => (n : Int))) current
  simpa [constants, List.map_map, List.mem_map, Function.comp_def] using tested

theorem insert_fresh (positions : List Nat) (current : Nat) (fresh : current ∉ positions) :
    ScalarSet.insert (constants positions) (.int current) = constants (positions ++ [current]) := by
  simp only [ScalarSet.insert, contains, show decide (current ∈ positions) = false by simp [fresh], Bool.false_eq_true, if_false]
  simp [constants]

variable {globals : Mutable.Globals}

theorem position_test (base : Mutable.Heap) (path positions : List Nat) (garbage : Mutable.Heap)
    (locals : HeapCalls.Env) (current : Nat)
    (getPositions : locals "positions" = some (.ref (base.length + 1)))
    (getCurrent : locals "current" = some (number current)) :
    Evaluates program locals (.membership (.name "current") (.name "positions") true)
      (work base path positions garbage) (.scalar (.bool (!decide (current ∈ positions))))
      (work base path positions garbage) globals := by
  apply Evaluates.membership (.cons (Evaluates.lookup getCurrent) (.cons (Evaluates.lookup getPositions) .nil))
  simpa only [contains] using membership_set (work base path positions garbage) (base.length + 1)
    (.int current) (constants positions) (positions_found base path positions garbage) true

theorem position_add (base : Mutable.Heap) (path positions : List Nat) (garbage : Mutable.Heap)
    (locals : HeapCalls.Env) (current : Nat) (fresh : current ∉ positions)
    (getPositions : locals "positions" = some (.ref (base.length + 1)))
    (getCurrent : locals "current" = some (number current)) :
    Evaluates program locals (.call (.attribute (.name "positions") "add") [.name "current"])
      (work base path positions garbage) (.scalar .none)
      (work base path (positions ++ [current]) garbage) globals := by
  have added := Evaluates.add (module := program) (globals := globals)
    (Evaluates.lookup getPositions) (positions_found base path positions garbage)
    (Evaluates.lookup getCurrent) (positions_found base path positions garbage)
  simpa only [insert_fresh positions current fresh, set_positions] using added

theorem path_append (base : Mutable.Heap) (path positions : List Nat) (garbage : Mutable.Heap)
    (locals : HeapCalls.Env) (current : Nat)
    (getPath : locals "path" = some (.ref base.length))
    (getCurrent : locals "current" = some (number current)) :
    Evaluates program locals (.call (.attribute (.name "path") "append") [.name "current"])
      (work base path positions garbage) (.scalar .none)
      (work base (path ++ [current]) positions garbage) globals := by
  have appended := Evaluates.append (module := program) (globals := globals)
    (Evaluates.lookup getPath) (path_found base path positions garbage)
    (Evaluates.lookup getCurrent) (path_found base path positions garbage)
  have same : numbers path ++ [number current] = numbers (path ++ [current]) := by simp
  simpa only [same, set_path] using appended

end solvers.«92».heap
