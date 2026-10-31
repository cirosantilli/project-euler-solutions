import solvers.«83.py.lean.d».InitialState

namespace solvers.«83»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RectangularMatrixCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
set_option Elab.async false

def solveBody : List Stmt := ((findFunctionIn program.body "solve").getD ([], [])).2
def loopTail : List Stmt := (executableBody solveBody).drop 13
def entryEnv (address : Nat) : HeapCalls.Env := argumentEnv arguments [.ref address]
def sizedEnv (address height width : Nat) : HeapCalls.Env :=
  setLocal (setLocal (entryEnv address) "n" (number height)) "m" (number width)
def totalEnv (address height width : Nat) : HeapCalls.Env :=
  setLocal (sizedEnv address height width) "total" (number (height * width))

theorem solve_source : findFunctionIn program.body "solve" = some (arguments, solveBody) := rfl

theorem entry_source : executableBody solveBody =
    [.ifThen (.unaryOp .not (.name "matrix")) [.return (.constant (.int 0))] [],
     .assert (.call (.name "is_rectangular") [.name "matrix"]) none,
     .assign [.name "n"] (.builtinCall .len [.name "matrix"]),
     .assign [.name "m"] (.builtinCall .len [.subscript (.name "matrix") (.constant (.int 0))]),
     .ifThen (.compare (.name "m") [(.eq, .constant (.int 0))]) [.return (.constant (.int 0))] [],
     .assign [.name "total"] (.binOp (.name "n") .mult (.name "m"))] ++ setupCode ++ loopTail := rfl

theorem matrix_height (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (localMatrix : env "matrix" = some (.ref address)) :
    Evaluates program env (.builtinCall .len [.name "matrix"]) heap (number rows.length) heap := by
  apply Evaluates.builtin (.cons (Evaluates.lookup localMatrix) .nil)
  simp [HeapCalls.primitive, Mutable.length, found]

theorem matrix_width (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (first : Nat × List Int) (rest : MatrixRows)
    (found : heap[address]? = some (.list ((first :: rest).map (fun pair => .ref pair.1))))
    (readFirst : heap[first.1]? = some (.list (encode first.2)))
    (localMatrix : env "matrix" = some (.ref address)) :
    Evaluates program env (.builtinCall .len [.subscript (.name "matrix") (.constant (.int 0))])
      heap (number first.2.length) heap := by
  have row : Evaluates program env (.subscript (.name "matrix") (.constant (.int 0))) heap (.ref first.1) heap := by
    apply Evaluates.subscript (by intros a b c impossible; cases impossible) (Evaluates.lookup localMatrix) Evaluates.constant
    simpa using Mutable.getItem_list heap address 0 ((first :: rest).map (fun pair => .ref pair.1)) found (by simp)
  apply Evaluates.builtin (.cons row .nil)
  simp [HeapCalls.primitive, Mutable.length, readFirst, encode]

theorem empty_result (heap : Mutable.Heap) (address : Nat) (found : heap[address]? = some (.list [])) :
    ReturnsValue program "solve" [.ref address] heap (number 0) heap := by
  have tested : Evaluates program (entryEnv address) (.unaryOp .not (.name "matrix")) heap
      (.scalar (.bool true)) heap :=
    Evaluates.unary (Evaluates.lookup rfl) (by simp [HeapCalls.unary, Mutable.truth, found])
  apply BlockReturns.function solve_source rfl
  rw [entry_source]
  apply BlockReturns.branch (flag := true) tested rfl
  exact BlockReturns.return_value Evaluates.constant

/-- Once the loop continuation is checked, this supplies the complete initialized call. -/
theorem positive_result_of_loop (heap after : Mutable.Heap) (address : Nat)
    (first : Nat × List Int) (rest : MatrixRows) (answer : Mutable.Value)
    (found : heap[address]? = some (.list ((first :: rest).map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ first :: rest, heap[pair.1]? = some (.list (encode pair.2)))
    (positive : 0 < first.2.length)
    (checked : ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap)
    (loop : let bound := ProjectEulerStatements.P83.bigVal ((first :: rest).map Prod.snd)
      let size := (rest.length + 1) * first.2.length
      BlockReturns program (initialEnv (totalEnv address (rest.length + 1) first.2.length) heap size bound)
        loopTail (initialHeap heap size bound first.2[0]) answer after) :
    ReturnsValue program "solve" [.ref address] heap answer after := by
  let tall := setLocal (entryEnv address) "n" (number (rest.length + 1))
  have tested : Evaluates program (entryEnv address) (.unaryOp .not (.name "matrix")) heap
      (.scalar (.bool false)) heap :=
    Evaluates.unary (Evaluates.lookup rfl) (by simp [HeapCalls.unary, Mutable.truth, found])
  have height := matrix_height heap (entryEnv address) address (first :: rest) found rfl
  have width := matrix_width heap tall address first rest found (reads first (by simp)) rfl
  have testedWidth : Evaluates program (sizedEnv address (rest.length + 1) first.2.length)
      (.compare (.name "m") [(.eq, .constant (.int 0))]) heap (.scalar (.bool false)) heap := by
    apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, show first.2.length ≠ 0 by omega]
  have size : Evaluates program (sizedEnv address (rest.length + 1) first.2.length)
      (.binOp (.name "n") .mult (.name "m")) heap
      (number ((rest.length + 1) * first.2.length)) heap := by
    apply Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup rfl)
    simp [Mutable.binary, scalarBinary, number]
  have allocated := setup_result heap (totalEnv address (rest.length + 1) first.2.length)
    address ((rest.length + 1) * first.2.length) first rest found reads positive (by positivity) rfl rfl
  apply BlockReturns.function solve_source rfl
  rw [entry_source]
  apply BlockReturns.branch (flag := false) tested rfl
  apply BlockReturns.assertion (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) checked) rfl
  apply BlockReturns.assign height
  apply BlockReturns.assign width
  apply BlockReturns.branch (flag := false) testedWidth rfl
  apply BlockReturns.assign size
  exact allocated.returns loop

theorem zero_width_result (heap : Mutable.Heap) (address : Nat) (first : Nat × List Int) (rest : MatrixRows)
    (found : heap[address]? = some (.list ((first :: rest).map (fun pair => .ref pair.1))))
    (readFirst : heap[first.1]? = some (.list (encode first.2))) (zero : first.2.length = 0)
    (checked : ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap) :
    ReturnsValue program "solve" [.ref address] heap (number 0) heap := by
  let tall := setLocal (entryEnv address) "n" (number (rest.length + 1))
  have tested : Evaluates program (entryEnv address) (.unaryOp .not (.name "matrix")) heap
      (.scalar (.bool false)) heap :=
    Evaluates.unary (Evaluates.lookup rfl) (by simp [HeapCalls.unary, Mutable.truth, found])
  have height := matrix_height heap (entryEnv address) address (first :: rest) found rfl
  have width := matrix_width heap tall address first rest found readFirst rfl
  have testedWidth : Evaluates program (sizedEnv address (rest.length + 1) first.2.length)
      (.compare (.name "m") [(.eq, .constant (.int 0))]) heap (.scalar (.bool true)) heap := by
    apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero]
  apply BlockReturns.function solve_source rfl
  rw [entry_source]
  apply BlockReturns.branch (flag := false) tested rfl
  apply BlockReturns.assertion (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) checked) rfl
  apply BlockReturns.assign height
  apply BlockReturns.assign width
  apply BlockReturns.branch (flag := true) testedWidth rfl
  exact BlockReturns.return_value Evaluates.constant

end solvers.«83»
