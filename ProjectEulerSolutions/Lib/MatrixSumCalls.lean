import ProjectEulerSolutions.Lib.RectangularMatrixCalls
import lang.py.FormalSW.SumCalls

/-! Nested sums over shared matrix rows preserve the entire caller heap. -/
namespace ProjectEulerSolutions.MatrixSumCalls
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false
variable {program : Module}

abbrev rowSum : Expr := .sumSequence (.name "row") (.constant (.int 0))
abbrev matrixSum : Expr := .sumGenerator rowSum [("row", .name "matrix", [])] (.constant (.int 0))
abbrev boundCode : List Stmt := [.return (.binOp matrixSum .add (.constant (.int 1)))]

theorem sum_result (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat) (rows : MatrixRows)
    (localMatrix : env "matrix" = some (.ref address))
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) :
    Evaluates program env matrixSum heap
      (.scalar (.int (rows.foldl (fun total pair => total + pair.2.foldl Int.add 0) 0))) heap := by
  apply Evaluates.sum_generator_sequence (items := rows.map (fun pair => .ref pair.1))
    (by intros expressions impossible; cases impossible) (Evaluates.lookup localMatrix)
    (by simp [Mutable.sequence, found]) Evaluates.constant rfl
  have loop := sum_map_loop (globals := []) (module := program) (heap := heap)
    (values := rows.attach) (encode := fun pair => .ref pair.val.1)
    (number := fun pair => pair.val.2.foldl Int.add 0)
    (name := "row") (element := rowSum) (source := .ref address) (position := 0)
    (env := setLocal (comprehensionEnv env [("row", .name "matrix", [])]) sumAccumulator (.scalar (.int 0)))
    (initial := 0) rows.attach (by decide)
    (by
      have mapped : rows.attach.map (fun pair => Mutable.Value.ref pair.val.1) =
          rows.map (fun pair => Mutable.Value.ref pair.1) := by
        exact List.attach_map_val (l := rows) (f := fun pair => Mutable.Value.ref pair.1)
      simp [sequenceValue, Mutable.sequence, found, mapped])
    (fun current pair lookup => Evaluates.sum_integers
      (.cons (Evaluates.lookup lookup) (.cons Evaluates.constant .nil))
      (by simp [Mutable.sequence, reads pair.val pair.property, encode]))
    rfl (by simp [setLocal])
  rw [List.foldl_attach (l := rows) (f := fun total pair => total + pair.2.foldl Int.add 0) (b := 0)] at loop
  simpa only [comprehensionFilters, sequenceSource] using loop

theorem bound_result
    (source : findFunctionIn program.body "big_val" = some (arguments, boundCode))
    (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) :
    ReturnsValue program "big_val" [.ref address] heap
      (.scalar (.int (rows.foldl (fun total pair => total + pair.2.foldl Int.add 0) 0 + 1))) heap := by
  apply BlockReturns.function source rfl
  apply BlockReturns.return_value
  exact Evaluates.binary (sum_result heap _ address rows rfl found reads) Evaluates.constant rfl

end ProjectEulerSolutions.MatrixSumCalls
