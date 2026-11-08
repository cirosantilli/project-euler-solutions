import solvers.«96.py.lean.d».Normalization
import ProjectEulerSolutions.Lib.SudokuText
import ProjectEulerSolutions.Lib.HeapCollection

namespace solvers.«96».parse_puzzles
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
abbrev rows (board : Board) : List Mutable.Value := (Encoding.rows board).map HeapTextLines.textValue
def references (root count : Nat) : List Mutable.Value :=
  (List.range count).map fun index => .ref (root + 1 + index)
def work (base : Mutable.Heap) (boards : List Board) : Mutable.Heap :=
  HeapCollection.working base (references base.length boards.length) (boards.map fun board => .list (rows board))
def Correct (board solved : Board) : Prop := Completion board solved ∧ ∀ final, Completion board final → final = solved

theorem root (base : Mutable.Heap) (boards : List Board) :
    (work base boards)[base.length]? = some (.list (references base.length boards.length)) := HeapCollection.found _ _ _

theorem length (base : Mutable.Heap) (boards : List Board) : (work base boards).length = base.length + 1 + boards.length := by
  simp [work, HeapCollection.working]
  omega

theorem frame (base : Mutable.Heap) (boards : List Board) (index : Nat) (inside : index < base.length) :
    (work base boards)[index]? = base[index]? := List.getElem?_append_left inside

theorem append_record (base : Mutable.Heap) (boards : List Board) (board : Board) :
    (work base boards ++ [Mutable.Object.list (rows board)]).set base.length
        (.list (references base.length boards.length ++ [Mutable.Value.ref (work base boards).length])) = work base (boards ++ [board]) := by
  simp [work, references, HeapCollection.working, List.set_append_right, List.range_succ, List.append_assoc, Nat.add_assoc]
  omega

theorem grid (base : Mutable.Heap) (boards : List Board) (index : Fin boards.length) :
    (work base boards)[base.length + 1 + index.val]? = some (.list (rows boards[index])) := by
  have offset : base.length + 1 + index.val - base.length = index.val + 1 := by omega
  simp only [work, HeapCollection.working, List.getElem?_append_right (show base.length ≤ base.length + 1 + index.val by omega),
    offset, List.getElem?_cons_succ, List.getElem?_map, List.getElem?_eq_getElem index.isLt, Option.map_some]
  rfl

abbrev definition := (findFunctionIn program.body "parse_puzzles").getD ([], [])
abbrev active : Expr := .compare (.name "i") [(.lt, .builtinCall .len [.name "lines"])]
abbrev header : Expr := .call (.attribute (.subscript (.name "lines") (.name "i")) "startswith") [.constant (.string "Grid")]
abbrev rowSlice : Expr := .subscript (.name "lines")
  (.slice (some (.binOp (.name "i") .add (.constant (.int 1))))
    (some (.binOp (.name "i") .add (.constant (.int 10)))) none)
abbrev collect : List Stmt := [.assign [.name "grid"] rowSlice,
  .augAssign (.name "i") .add (.constant (.int 10)),
  .expr (.call (.attribute (.name "puzzles") "append") [.name "grid"])]
abbrev body : List Stmt := [.ifThen header collect [.augAssign (.name "i") .add (.constant (.int 1))]]
abbrev loop : Stmt := .whileDo active body []
abbrev finish : Stmt := .return (.name "puzzles")

theorem source : definition.2 = [.assign [.name "lines"] normalization,
    .assign [.name "puzzles"] (.list []), .assign [.name "i"] (.constant (.int 0)), loop, finish] := rfl

structure Locals (address root position : Nat) (env : HeapCalls.Env) : Prop where
  lines : env "lines" = some (.ref address)
  puzzles : env "puzzles" = some (.ref root)
  index : env "i" = some (number position)

theorem measure (lines : List String) (position address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (found : heap[address]? = some (.list (lines.map HeapTextLines.textValue)))
    (getLines : env "lines" = some (.ref address)) (getPosition : env "i" = some (number position)) :
    Evaluates program env active heap (.scalar (.bool (decide (position < lines.length)))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (bv := number lines.length) (Evaluates.lookup getPosition)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getLines) .nil)
    simp [primitive, Mutable.length, found]
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem read_header (lines : List String) (position address : Nat) (line : String)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list (lines.map HeapTextLines.textValue)))
    (getLines : env "lines" = some (.ref address)) (getPosition : env "i" = some (number position))
    (atPosition : lines[position]? = some line) :
    Evaluates program env header heap (.scalar (.bool (Strings.startsWith line "Grid"))) heap globals := by
  have indexed : Evaluates program env (.subscript (.name "lines") (.name "i")) heap
      (.scalar (.string line)) heap globals := by
    apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getLines) (Evaluates.lookup getPosition)
    have bound := (List.getElem?_eq_some_iff.mp atPosition).1
    simp [Mutable.getItem, Mutable.index, found, atPosition, number, Except.map,
      not_lt.mpr (Int.natCast_nonneg position), not_le.mpr bound]
  exact Evaluates.starts_with indexed Evaluates.constant (by intro name impossible; cases impossible)

theorem slice_result (lines : List String) (position address : Nat) (board : Board)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list (lines.map HeapTextLines.textValue)))
    (getLines : env "lines" = some (.ref address)) (getPosition : env "i" = some (number position))
    (selected : (lines.drop (position + 1)).take 9 = Encoding.rows board) :
    Evaluates program env rowSlice heap (.ref heap.length) (heap ++ [.list (rows board)]) globals := by
  have size := congrArg List.length selected
  simp only [List.length_take, List.length_drop, Encoding.rows, List.length_ofFn] at size
  have bound : position + 10 ≤ (lines.map HeapTextLines.textValue).length := by simp; omega
  apply Evaluates.slice (middle := heap) (first := number (position + 1)) (last := number (position + 10))
  · apply Evaluates.Arguments.cons (middle := heap) (Evaluates.lookup getLines)
    apply Evaluates.Arguments.cons (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap) (Evaluates.lookup getPosition) Evaluates.constant
      simp [Mutable.binary, scalarBinary, number]
    apply Evaluates.Arguments.cons (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap) (Evaluates.lookup getPosition) Evaluates.constant
      simp [Mutable.binary, scalarBinary, number]
    exact .cons Evaluates.constant .nil
  · have sliced := Mutable.getSlice_list_interval heap address (lines.map HeapTextLines.textValue)
      (position + 1) (position + 10) (by omega) bound found
    simpa only [show position + 10 - (position + 1) = 9 by omega, ← List.map_drop, ← List.map_take, selected] using sliced

theorem collect_result (lines : List String) (position address : Nat) (board : Board)
    (base : Mutable.Heap) (boards : List Board) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : (work base boards)[address]? = some (.list (lines.map HeapTextLines.textValue)))
    (locals : Locals address base.length position env)
    (selected : (lines.drop (position + 1)).take 9 = Encoding.rows board) :
    BlockContinues program env collect (work base boards)
      (setLocal (setLocal env "grid" (.ref (work base boards).length)) "i" (number (position + 10)))
      (work base (boards ++ [board])) globals := by
  let withGrid := setLocal env "grid" (.ref (work base boards).length)
  let ready := setLocal withGrid "i" (number (position + 10))
  let sliced := work base boards ++ [Mutable.Object.list (rows board)]
  have prepared := BlockContinues.assign (name := "grid")
    (slice_result lines position address board (work base boards) env globals found locals.lines locals.index selected)
  have incremented : BlockContinues program withGrid [.augAssign (.name "i") .add (.constant (.int 10))]
      sliced ready sliced globals := by
    apply BlockContinues.augAssign (by simpa [withGrid, setLocal] using locals.index) Evaluates.constant
    simp [inplace, Mutable.binary, scalarBinary, number]
  have getPuzzles : ready "puzzles" = some (.ref base.length) := by
    simpa [ready, withGrid, setLocal] using locals.puzzles
  have foundPuzzles := HeapCollection.prefix_found (work base boards) [Mutable.Object.list (rows board)]
    base.length _ (root base boards)
  have appended := Evaluates.append (module := program) (globals := globals) (Evaluates.lookup getPuzzles)
    foundPuzzles (Evaluates.lookup (show ready "grid" = some (.ref (work base boards).length) by simp [ready, withGrid, setLocal]))
    foundPuzzles
  have finished := BlockContinues.discard appended
  rw [append_record] at finished
  exact prepared.append (incremented.append finished)

theorem iteration (lines : List String) (position address : Nat) (heap after : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (answer : Mutable.Value)
    (found : heap[address]? = some (.list (lines.map HeapTextLines.textValue)))
    (getLines : env "lines" = some (.ref address)) (getPosition : env "i" = some (number position))
    (inside : position < lines.length)
    (executed : BlockReturns program env (body ++ [loop, finish]) heap answer after globals) :
    BlockReturns program env [loop, finish] heap answer after globals := by
  apply BlockReturns.while_loop (flag := true) (measure lines position address heap env globals found getLines getPosition)
    (by simp [Mutable.truth, scalarTruth, inside])
  have controlled : Calls.loopControl body = false := rfl
  simpa only [Calls.loopContinuation, controlled, Bool.false_eq_true, if_false, List.append_assoc] using executed

end
end solvers.«96».parse_puzzles
