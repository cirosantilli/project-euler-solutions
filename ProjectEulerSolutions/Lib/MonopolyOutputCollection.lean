import ProjectEulerSolutions.Lib.MonopolyOutputSlices

namespace ProjectEulerSolutions.MonopolyOutput
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def squareRange (values : Fin 120 → Int) (first count : Nat) : List Mutable.Value :=
  (List.range' first count).map fun index => .scalar (.int (Monopoly.squareWeight values (Fin.ofNat 40 index)))

theorem squareRange_full (values : Fin 120 → Int) : squareRange values 0 40 = encode (Monopoly.squareWeight values) := by
  apply List.ext_getElem
  · simp only [squareRange, encode, List.length_map, List.length_range', List.length_ofFn]
  · intro index left right
    have inside : index < 40 := by simpa only [encode, List.length_ofFn] using right
    simp only [encode, List.getElem_ofFn]
    simp [squareRange, Fin.ofNat, Nat.mod_eq_of_lt inside]

theorem squares_scan (module : Module) (values : Fin 120 → Int) (base : Mutable.Heap) (address : Nat)
    (found : base[address]? = some (.list (encode values)))
    (count first stop : Nat) (length : first + count = stop) (bound : stop ≤ 40)
    (output : List Mutable.Value) (garbage : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getWeights : env "weights" = some (.ref address)) :
    ∃ afterEnv extra, BlockContinues module env
      [.rangeStepLoop "pos" first stop 1 [.comprehensionAppend base.length squareSum] []]
      (HeapCollection.working base output garbage) afterEnv
      (HeapCollection.working base (output ++ squareRange values first count) extra) globals := by
  induction count generalizing first output garbage env with
  | zero =>
      refine ⟨env, garbage, ?_⟩
      simp only [squareRange, List.range'_zero, List.map_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      have inside : first < 40 := by omega
      let square : Fin 40 := Fin.ofNat 40 first
      let boundEnv := setLocal env "pos" (.scalar (.int first))
      have evaluated := sum_result module values square (HeapCollection.working base output garbage) address
        (HeapCollection.prefix_found base _ address _ found) boundEnv globals
        (by simpa [boundEnv, setLocal] using getWeights)
        (by simp [boundEnv, setLocal, square, Nat.mod_eq_of_lt inside])
      have emitted := HeapCollection.emit base output garbage
        [.list (encode fun doubles : Fin 3 => values (Monopoly.stateIndex (square, doubles)))] _ evaluated
      obtain ⟨afterEnv, extra, rest⟩ := ih (first + 1) (by omega)
        (output ++ [.scalar (.int (Monopoly.squareWeight values square))])
        (garbage ++ [.list (encode fun doubles : Fin 3 => values (Monopoly.stateIndex (square, doubles)))]) boundEnv
        (by simpa [boundEnv, setLocal] using getWeights)
      refine ⟨afterEnv, extra, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true,
        if_false, squareRange, List.range'_succ, List.map_cons, List.append_assoc, List.singleton_append,
        Int.natCast_add, Int.natCast_one] using emitted.append rest

def squareCollection : Expr := .listComp squareSum [("pos", .call (.name "range") [.constant (.int 40)], [])]

/-- The actual comprehension allocates the forty sums in board order and preserves the input weights. -/
theorem squares_result (module : Module) (values : Fin 120 → Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode values))) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getWeights : env "weights" = some (.ref address)) :
    ∃ extra, Evaluates module env squareCollection heap (.ref heap.length)
      (HeapCollection.working heap (encode (Monopoly.squareWeight values)) extra) globals := by
  obtain ⟨afterEnv, extra, ran⟩ := squares_scan module values heap address found 40 0 40 rfl (Nat.le_refl _) [] []
    (comprehensionEnv env [("pos", .call (.name "range") [.constant (.int 40)], [])]) globals
    (by simpa [comprehensionEnv] using getWeights)
  refine ⟨extra, ?_⟩
  apply Evaluates.list_comp_range (.cons Evaluates.constant .nil) rfl
  simpa only [squareRange_full, List.nil_append, comprehensionFilters, comprehensionBody, HeapCollection.working] using ran

end ProjectEulerSolutions.MonopolyOutput
