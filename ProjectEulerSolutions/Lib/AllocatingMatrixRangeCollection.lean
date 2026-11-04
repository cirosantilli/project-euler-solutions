import ProjectEulerSolutions.Lib.MatrixRangeCollection

/-! Matrix comprehensions preserve caller objects while row calls allocate private storage. -/
namespace ProjectEulerSolutions.AllocatingMatrixRangeCollection
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.TriangleBufferCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
set_option Elab.async false

def state (base : Mutable.Heap) (rows : MatrixRows) (privateHeap : Mutable.Heap) : Mutable.Heap :=
  base ++ [.list (rows.map (fun row => .ref row.1))] ++ privateHeap

theorem prefix_found (base privateHeap : Mutable.Heap) (rows : MatrixRows) (address : Nat) (object : Mutable.Object)
    (found : base[address]? = some object) : (state base rows privateHeap)[address]? = some object := by
  unfold state
  rw [List.append_assoc, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

theorem advance (base privateHeap extra : Mutable.Heap) (rows : MatrixRows) (values : List Int) :
    (((state base rows privateHeap) ++ [Mutable.Object.list (encode values)]) ++ extra).set base.length
      (Mutable.Object.list (rows.map (fun row => Mutable.Value.ref row.1) ++ [Mutable.Value.ref (state base rows privateHeap).length])) =
      state base (rows ++ [((state base rows privateHeap).length, values)]) (privateHeap ++ [.list (encode values)] ++ extra) := by
  simp [state, List.append_assoc]

theorem advance_read (base privateHeap extra : Mutable.Heap) (rows : MatrixRows) (values : List Int)
    (address : Nat) (object : Mutable.Object) (different : address ≠ base.length)
    (found : (state base rows privateHeap)[address]? = some object) :
    (state base (rows ++ [((state base rows privateHeap).length, values)])
      (privateHeap ++ [.list (encode values)] ++ extra))[address]? = some object := by
  rw [← advance, List.getElem?_set_ne (Ne.symm different), List.append_assoc,
    List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

theorem advance_row (base privateHeap extra : Mutable.Heap) (rows : MatrixRows) (values : List Int) :
    (state base (rows ++ [((state base rows privateHeap).length, values)])
      (privateHeap ++ [.list (encode values)] ++ extra))[(state base rows privateHeap).length]? = some (.list (encode values)) := by
  have different : (state base rows privateHeap).length ≠ base.length := by simp [state]
  rw [← advance, List.getElem?_set_ne (Ne.symm different), List.append_assoc]
  simp

theorem collect_loop (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element : Expr) (values : Nat → List Int) (cursor height : Nat)
    (output : MatrixRows) (privateHeap : Mutable.Heap)
    (bounded : cursor ≤ height)
    (outputValues : output.map Prod.snd = (List.range cursor).map values)
    (outputReads : ∀ row ∈ output, (state base output privateHeap)[row.1]? = some (.list (encode row.2)))
    (outputPrivate : ∀ row ∈ output, base.length < row.1)
    (evaluate : ∀ (collected : MatrixRows) (objects : Mutable.Heap) (index : Nat), index < height →
      ∃ extra, Evaluates program (setLocal env name (number index)) element (state base collected objects)
        (.ref (state base collected objects).length)
        (((state base collected objects) ++ [.list (encode (values index))]) ++ extra)) :
    ∃ (after : HeapCalls.Env) (extra : Mutable.Heap) (result : MatrixRows),
      BlockContinues program env [.rangeStepLoop name cursor height 1 [.comprehensionAppend base.length element] []]
        (state base output privateHeap) after (state base result (privateHeap ++ extra)) ∧
      result.map Prod.snd = (List.range height).map values ∧
      (∀ row ∈ result, (state base result (privateHeap ++ extra))[row.1]? = some (.list (encode row.2))) ∧
      (∀ row ∈ result, base.length < row.1) := by
  by_cases active : cursor < height
  · let entered := setLocal env name (number cursor)
    let newRow := ((state base output privateHeap).length, values cursor)
    obtain ⟨allocated, evaluated⟩ := evaluate output privateHeap cursor active
    have collector : (((state base output privateHeap) ++ [.list (encode (values cursor))]) ++ allocated)[base.length]? =
        some (.list (output.map (fun row => .ref row.1))) := by simp [state, List.append_assoc]
    have emitted := BlockContinues.comprehension_append evaluated collector
    rw [advance] at emitted
    have newValues : (output ++ [newRow]).map Prod.snd = (List.range (cursor + 1)).map values := by
      simp [newRow, outputValues, List.range_succ]
    have newReads : ∀ row ∈ output ++ [newRow],
        (state base (output ++ [newRow]) (privateHeap ++ [.list (encode (values cursor))] ++ allocated))[row.1]? =
          some (.list (encode row.2)) := by
      intro row member
      rcases List.mem_append.mp member with old | fresh
      · exact advance_read base privateHeap allocated output (values cursor) row.1 _
          (by have := outputPrivate row old; omega) (outputReads row old)
      · have equal : row = newRow := by simpa only [List.mem_singleton] using fresh
        subst row
        exact advance_row base privateHeap allocated output (values cursor)
    have newPrivate : ∀ row ∈ output ++ [newRow], base.length < row.1 := by
      intro row member
      rcases List.mem_append.mp member with old | fresh
      · exact outputPrivate row old
      · have equal : row = newRow := by simpa only [List.mem_singleton] using fresh
        subst row
        simp [newRow, state]
    obtain ⟨after, more, result, continued, data, reads, owned⟩ := collect_loop program base entered name element values
      (cursor + 1) height (output ++ [newRow]) (privateHeap ++ [.list (encode (values cursor))] ++ allocated)
      (by omega) newValues newReads newPrivate (by
        intro collected objects index inside
        have same : setLocal entered name (number index) = setLocal env name (number index) := by
          funext query
          by_cases equal : query = name <;> simp [entered, setLocal, equal]
        rw [same]
        exact evaluate collected objects index inside)
    refine ⟨after, [.list (encode (values cursor))] ++ allocated ++ more, result, ?_, data, ?_, owned⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [newRow, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Bool.false_or, Bool.false_eq_true, if_false, List.append_assoc, Int.natCast_add, Int.natCast_one] using
        emitted.append continued
    · simpa only [List.append_assoc] using reads
  · have done : cursor = height := by omega
    subst cursor
    refine ⟨env, [], output, ?_, outputValues, ?_, outputPrivate⟩
    · simpa only [List.append_nil] using
        (BlockContinues.range_done (module := program) (env := env) (heap := state base output privateHeap)
          (by simp [Calls.rangeContinues]) BlockContinues.nil)
    · simpa only [List.append_nil] using outputReads
termination_by height - cursor
decreasing_by omega

theorem collect (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element : Expr) (values : Nat → List Int) (height : Nat)
    (expressions : List Expr) (args : List Mutable.Value)
    (evaluated : Evaluates.Arguments program env expressions base args base)
    (bounds : rangeBounds args = .ok (0, height, 1))
    (evaluate : ∀ (collected : MatrixRows) (objects : Mutable.Heap) (index : Nat), index < height →
      ∃ extra, Evaluates program
        (setLocal (comprehensionEnv env [(name, .call (.name "range") expressions, [])]) name (number index))
        element (state base collected objects) (.ref (state base collected objects).length)
        (((state base collected objects) ++ [.list (encode (values index))]) ++ extra)) :
    ∃ (extra : Mutable.Heap) (result : MatrixRows),
      Evaluates program env (.listComp element [(name, .call (.name "range") expressions, [])]) base (.ref base.length)
        (state base result extra) ∧ result.map Prod.snd = (List.range height).map values ∧
      (∀ row ∈ result, (state base result extra)[row.1]? = some (.list (encode row.2))) := by
  obtain ⟨after, extra, result, loop, data, reads, _⟩ := collect_loop program base
    (comprehensionEnv env [(name, .call (.name "range") expressions, [])]) name element values 0 height [] []
    (by omega) rfl (by simp) (by simp) evaluate
  refine ⟨extra, result, ?_, data, ?_⟩
  · apply Evaluates.list_comp_range evaluated bounds
    simpa only [state, List.map_nil, List.append_nil, List.nil_append, comprehensionBody, comprehensionFilters] using loop
  · simpa only [List.nil_append] using reads

end ProjectEulerSolutions.AllocatingMatrixRangeCollection
