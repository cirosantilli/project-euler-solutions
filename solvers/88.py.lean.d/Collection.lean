import solvers.«88.py.lean.d».Preparation
import ProjectEulerSolutions.Lib.ProductSumDistinct
import lang.py.FormalSW.ScalarSetLemmas
import lang.py.FormalSW.SetSumCalls

namespace solvers.«88».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

def minimaValues (maxSize : Nat) : List Nat := (table maxSize).drop 2
def setKeys (maxSize : Nat) : List Constant := (minimaValues maxSize).eraseDups.map (fun (n : Nat) => .int n)
def setHeap (context : Context) (maxSize : Nat) : Mutable.Heap :=
  working context (table maxSize) ++ [.list (encode (minimaValues maxSize)), .set (setKeys maxSize)]
abbrev suffixExpression : Expr := .subscript tableExpression (.slice (some (.constant (.int 2))) none none)

theorem slice_result (maxSize : Nat) (context : Context) (cells : Cells context maxSize (2 * maxSize))
    (env : HeapCalls.Env) (getCell : env "__formalsw_cell_solve_best" = some (.ref context.bestCell)) :
    Evaluates program env suffixExpression (working context (table maxSize))
      (.ref (working context (table maxSize)).length)
      (working context (table maxSize) ++ [.list (encode (minimaValues maxSize))]) := by
  apply Evaluates.slice (receiver := .ref context.base.length)
    (first := number 2) (last := .scalar .none) (stepValue := .scalar .none)
    (middle := working context (table maxSize))
  · exact .cons (captured_table (working context (table maxSize)) context.bestCell context.base.length env
        getCell (cells.best (table maxSize)))
      (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))
  · simpa only [encode, minimaValues, List.map_drop] using
      Mutable.getSlice_list_suffix (working context (table maxSize)) context.base.length (encode (table maxSize)) 2
        (working_found context (table maxSize))

/-- Slicing and set construction observe exactly the minima, with duplicates removed. -/
theorem set_result (maxSize : Nat) (context : Context) (cells : Cells context maxSize (2 * maxSize))
    (env : HeapCalls.Env) (getCell : env "__formalsw_cell_solve_best" = some (.ref context.bestCell)) :
    Evaluates program env (.builtinCall .set [suffixExpression]) (working context (table maxSize))
      (.ref ((working context (table maxSize)).length + 1)) (setHeap context maxSize) := by
  let heap := working context (table maxSize)
  let sliceHeap := heap ++ [.list (encode (minimaValues maxSize))]
  have found : sliceHeap[heap.length]? =
      some (.list (((minimaValues maxSize).map (fun (n : Nat) => Constant.int n)).map Mutable.Value.scalar)) := by
    simp [sliceHeap, encode, number, List.map_map]
  have constructed := Mutable.setFrom_list sliceHeap heap.length
    ((minimaValues maxSize).map (fun (n : Nat) => Constant.int n)) found
  rw [ScalarSet.fold_insert_nats] at constructed
  apply Evaluates.builtin (name := .set) (values := [.ref heap.length]) (middle := sliceHeap)
  · exact .cons (slice_result maxSize context cells env getCell) .nil
  · simpa [primitive, setHeap, setKeys, sliceHeap, heap, List.append_assoc] using constructed

theorem sum_result (maxSize : Nat) (context : Context) (env : HeapCalls.Env)
    (getSet : env "uniq" = some (.ref ((working context (table maxSize)).length + 1))) :
    SetCalls.Evaluates program env (.sumSequence (.name "uniq") (.constant (.int 0)))
      (setHeap context maxSize) (number (answer maxSize)) (setHeap context maxSize) := by
  have sumCast (items : List Nat) : (items.map (fun (n : Nat) => (n : Int))).sum = (items.sum : Int) := by
    induction items with
    | nil => rfl
    | cons head tail ih => simp only [List.map_cons, List.sum_cons, ih, Nat.cast_add]
  have found : (setHeap context maxSize)[(working context (table maxSize)).length + 1]? =
      some (.set (((minimaValues maxSize).eraseDups.map (fun (n : Nat) => (n : Int))).map Constant.int)) := by
    simp [setHeap, setKeys, List.map_map]
  have summed := SetCalls.Evaluates.sum_integers
    (items := (minimaValues maxSize).eraseDups.map (fun (n : Nat) => (n : Int)))
    (module := program) (env := env) (globals := []) (initial := 0)
    (iterable := .name "uniq") (start := .constant (.int 0))
    (.cons (Evaluates.lookup getSet) (.cons Evaluates.constant .nil)) found
  simpa only [Int.zero_add, sumCast, minimaValues, answer_table] using summed

end solvers.«88».solve
