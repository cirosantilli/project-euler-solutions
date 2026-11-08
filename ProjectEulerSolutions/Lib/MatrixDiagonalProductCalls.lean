import ProjectEulerSolutions.Lib.MutableIntegerMatrixAccess
import ProjectEulerSolutions.Lib.HeapRangeFold
import Mathlib.Algebra.BigOperators.Fin

/-! Source multiplication of every diagonal entry in an augmented integer matrix. -/
namespace ProjectEulerSolutions.MatrixDiagonalProductCalls
open FormalSW.PyAst HeapCalls
open scoped BigOperators
set_option Elab.async false
variable {n : Nat}

def diagonal (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (index : Nat) : Int :=
  if inside : index < n then matrix ⟨index, inside⟩ ⟨index, by omega⟩ else 0

def body : List Stmt :=
  [.augAssign (.name "scale") .mult (.subscript (.subscript (.name "matrix") (.name "row")) (.name "row"))]

theorem fold_product (matrix : Matrix (Fin n) (Fin (n + 1)) Int) :
    (List.range' 0 n).foldl (fun product index => product * diagonal matrix index) 1 =
      ∏ row, matrix row row.castSucc := by
  have mapped : (List.range' 0 n).map (diagonal matrix) = List.ofFn (fun row => matrix row row.castSucc) := by
    apply List.ext_getElem
    · simp only [List.length_map, List.length_range', List.length_ofFn]
    · intro index left right
      have inside : index < n := by simpa only [List.length_ofFn] using right
      simp only [List.getElem_map, List.getElem_range', Nat.one_mul, Nat.zero_add, List.getElem_ofFn, diagonal, dif_pos inside]
      rfl
  calc
    _ = ((List.range' 0 n).map (diagonal matrix)).foldl (· * ·) 1 := by rw [List.foldl_map]
    _ = (List.ofFn (fun row => matrix row row.castSucc)).prod := by rw [mapped, List.prod_eq_foldl]
    _ = ∏ row, matrix row row.castSucc := List.prod_ofFn

/-- The initialized range fold changes only the scale accumulator and row index, preserving the complete heap. -/
theorem result (module : Module) (matrix : Matrix (Fin n) (Fin (n + 1)) Int)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin n → Nat)
    (view : MutableIntegerMatrix.View matrix heap root addresses)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getSize : env "size" = some (.scalar (.int n))) :
    ∃ afterEnv, BlockContinues module env
      [.assign [.name "scale"] (.constant (.int 1)), .forIn (.name "row") (.call (.name "range") [.name "size"]) body []]
      heap afterEnv heap globals ∧
      afterEnv "scale" = some (.scalar (.int (∏ row, matrix row row.castSucc))) ∧
      HeapRangeFold.Frame ["row", "scale"] env afterEnv := by
  let initial := setLocal env "scale" (.scalar (.int 1))
  let invariant := fun current (locals : HeapCalls.Env) (product : Int) =>
    current = heap ∧ locals "matrix" = some (.ref root) ∧ locals "scale" = some (.scalar (.int product))
  obtain ⟨afterEnv, afterHeap, ran, safe, _, frame⟩ := HeapRangeFold.loop module "row" body rfl ["row", "scale"] (by simp)
    (fun product index => product * diagonal matrix index) invariant 0 n
    (by
      intro current locals product index safe _ _
      exact ⟨safe.1, by simpa [setLocal] using safe.2.1, by simpa [setLocal] using safe.2.2⟩)
    (by
      intro current locals product index safe _ inside getRow
      obtain ⟨same, getCurrentMatrix, getScale⟩ := safe
      subst current
      let row : Fin n := ⟨index, inside⟩
      have read := view.entry_result module locals globals (.subscript (.name "matrix") (.name "row")) (.name "row") row row.castSucc
        (view.row_result module locals globals (.name "matrix") (.name "row") row (.lookup getCurrentMatrix) (.lookup getRow) (by intros; simp))
        (.lookup getRow) (by intros; simp)
      refine ⟨setLocal locals "scale" (.scalar (.int (product * diagonal matrix index))), heap, ?_, ?_,
        ⟨[], by simp⟩, HeapRangeFold.Frame.updated _ _ _ _ (by simp)⟩
      · apply BlockContinues.augAssign getScale read
        simp only [HeapCalls.inplace, Mutable.binary, scalarBinary, diagonal, dif_pos inside]
        rfl
      · exact ⟨rfl, by simpa [setLocal] using getCurrentMatrix, rfl⟩)
    n 0 (Nat.le_refl _) (by omega) heap initial 1
    ⟨rfl, by simpa [initial, setLocal] using getMatrix, rfl⟩
  obtain ⟨same, _, getFinal⟩ := safe
  subst afterHeap
  have loop : BlockContinues module initial [.forIn (.name "row") (.call (.name "range") [.name "size"]) body []]
      heap afterEnv heap globals :=
    BlockContinues.for_range (.cons (.lookup (by simpa [initial, setLocal] using getSize)) .nil) rfl ran
  refine ⟨afterEnv, (BlockContinues.assign Evaluates.constant).append loop, ?_,
    (HeapRangeFold.Frame.updated _ _ _ _ (by simp)).trans frame⟩
  simpa only [fold_product] using getFinal

end ProjectEulerSolutions.MatrixDiagonalProductCalls
