import ProjectEulerSolutions.Lib.IntegerRangeCollection
import ProjectEulerSolutions.Lib.RectangularMatrixCalls

/-! Nested range comprehensions that allocate one integer list for each row. -/
namespace ProjectEulerSolutions.MatrixRangeCollection
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.TriangleBufferCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
set_option Elab.async false

def rowRefs (base : Mutable.Heap) (count : Nat) : List Mutable.Value :=
  (List.range count).map (fun index => .ref (base.length + 1 + index))
def matrixHeap (base : Mutable.Heap) (values : Nat → List Int) (count : Nat) : Mutable.Heap :=
  base ++ [.list (rowRefs base count)] ++
    (List.range count).map (fun index => .list (encode (values index)))
def rowPairs (base : Mutable.Heap) (values : Nat → List Int) (count : Nat) : MatrixRows :=
  (List.range count).map (fun index => (base.length + 1 + index, values index))

@[simp] theorem length_heap (base : Mutable.Heap) (values : Nat → List Int) (count : Nat) :
    (matrixHeap base values count).length = base.length + 1 + count := by simp [matrixHeap]; omega

theorem prefix_found (base : Mutable.Heap) (values : Nat → List Int) (count address : Nat) (object : Mutable.Object)
    (found : base[address]? = some object) : (matrixHeap base values count)[address]? = some object := by
  unfold matrixHeap
  rw [List.append_assoc, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

theorem advance (base : Mutable.Heap) (values : Nat → List Int) (count : Nat) :
    ((matrixHeap base values count) ++ [Mutable.Object.list (encode (values count))]).set base.length
      (Mutable.Object.list (rowRefs base count ++ [Mutable.Value.ref (matrixHeap base values count).length])) =
      matrixHeap base values (count + 1) := by
  simp [matrixHeap, rowRefs, List.range_succ, List.append_assoc]; omega

theorem matrix_found (base : Mutable.Heap) (values : Nat → List Int) (count : Nat) :
    (matrixHeap base values count)[base.length]? = some (.list ((rowPairs base values count).map (fun pair => .ref pair.1))) := by
  simp [matrixHeap, rowPairs, rowRefs, List.append_assoc, List.map_map, Function.comp_def]

theorem rows_found (base : Mutable.Heap) (values : Nat → List Int) (count : Nat) :
    ∀ pair ∈ rowPairs base values count,
      (matrixHeap base values count)[pair.1]? = some (.list (encode pair.2)) := by
  intro pair member
  obtain ⟨index, inside, rfl⟩ := List.mem_map.mp member
  have bound := List.mem_range.mp inside
  unfold matrixHeap
  rw [List.getElem?_append_right (by simp)]
  have item : (List.range count)[index]? = some index := by
    rw [List.getElem?_eq_getElem (by simpa using bound)]
    simp
  simp [item]

theorem collect_loop (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element : Expr) (values : Nat → List Int) (cursor height : Nat)
    (bounded : cursor ≤ height)
    (evaluate : ∀ index, index < height →
      Evaluates program (setLocal env name (number index)) element (matrixHeap base values index)
        (.ref (matrixHeap base values index).length)
        ((matrixHeap base values index) ++ [.list (encode (values index))])) :
    ∃ after, BlockContinues program env
      [.rangeStepLoop name cursor height 1 [.comprehensionAppend base.length element] []]
      (matrixHeap base values cursor) after (matrixHeap base values height) := by
  by_cases active : cursor < height
  · let entered := setLocal env name (number cursor)
    have found : ((matrixHeap base values cursor) ++ [.list (encode (values cursor))])[base.length]? =
        some (.list (rowRefs base cursor)) := by simp [matrixHeap, List.append_assoc]
    have emitted := BlockContinues.comprehension_append (evaluate cursor active) found
    rw [advance] at emitted
    obtain ⟨after, continued⟩ := collect_loop program base entered name element values (cursor + 1) height (by omega) (by
      intro index inside
      have same : setLocal entered name (number index) = setLocal env name (number index) := by
        funext query
        by_cases equal : query = name <;> simp [entered, setLocal, equal]
      rw [same]
      exact evaluate index inside)
    refine ⟨after, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true,
      if_false, Int.natCast_add, Int.natCast_one] using emitted.append continued
  · have done : cursor = height := by omega
    subst cursor
    exact ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil⟩
termination_by height - cursor
decreasing_by omega

theorem collect (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element : Expr) (values : Nat → List Int) (height : Nat)
    (expressions : List Expr) (args : List Mutable.Value)
    (evaluated : Evaluates.Arguments program env expressions base args base)
    (bounds : rangeBounds args = .ok (0, height, 1))
    (evaluate : ∀ index, index < height →
      Evaluates program
        (setLocal (comprehensionEnv env [(name, .call (.name "range") expressions, [])]) name (number index))
        element (matrixHeap base values index) (.ref (matrixHeap base values index).length)
        ((matrixHeap base values index) ++ [.list (encode (values index))])) :
    Evaluates program env (.listComp element [(name, .call (.name "range") expressions, [])])
      base (.ref base.length) (matrixHeap base values height) := by
  obtain ⟨after, loop⟩ := collect_loop program base
    (comprehensionEnv env [(name, .call (.name "range") expressions, [])]) name element values 0 height (by omega) evaluate
  apply Evaluates.list_comp_range evaluated bounds
  simpa [matrixHeap, rowRefs, comprehensionBody, comprehensionFilters] using loop

end ProjectEulerSolutions.MatrixRangeCollection
