import ProjectEulerSolutions.Lib.TriangleBufferCalls

namespace ProjectEulerSolutions.IntegerListMinimumCalls
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.TriangleBufferCalls
set_option Elab.async false

theorem minimumFrom_encode (values : List Int) (best : Int) :
    Calls.minimumFrom (.int best) (values.map Constant.int) = .ok (.int (values.foldl min best)) := by
  induction values generalizing best with
  | nil => rfl
  | cons head tail ih =>
      by_cases smaller : head < best
      · simp [Calls.minimumFrom, scalarCompare, evalCmpOp, smaller, ih,
          Int.min_eq_right (Int.le_of_lt smaller)]
      · simp [Calls.minimumFrom, scalarCompare, evalCmpOp, smaller, ih,
          Int.min_eq_left (Int.le_of_not_gt smaller)]

theorem minimum_evaluate (program : Module) (name : String) (first : Int) (rest : List Int) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (address : Nat) (localValue : env name = some (.ref address))
    (found : heap[address]? = some (.list (encode (first :: rest)))) :
    Evaluates program env (.builtinCall .min [.name name]) heap
      (number (rest.foldl min first)) heap := by
  apply Evaluates.builtin (.cons (Evaluates.lookup localValue) .nil)
  have converted (values : List Int) :
      ((encode values).mapM (fun item => match item with
        | .scalar value => pure value
        | .ref _ => Mutable.unsupported) : Mutable.Eval (List Constant)) = pure (values.map Constant.int) := by
    simp only [encode, List.mapM_map]
    change (values.mapM (fun value => pure (Constant.int value)) : Mutable.Eval (List Constant)) = _
    exact List.mapM_pure
  have execution (items : List Mutable.Value) (present : heap[address]? = some (.list items)) :
      primitive (.builtin .min) [.ref address] heap =
        ((do
          let scalars ← items.mapM (fun item => match item with
            | .scalar value => pure value
            | .ref _ => Mutable.unsupported)
          scalarResult (some (ScalarSet.minimum scalars))) : Mutable.Eval Mutable.Value) heap := by
    simp [primitive, Mutable.sequence, present]
    rfl
  rw [execution (encode (first :: rest)) found, converted]
  change scalarResult (some (ScalarSet.minimum ((first :: rest).map Constant.int))) heap = _
  rw [show ScalarSet.minimum ((first :: rest).map Constant.int) =
    .ok (.int (rest.foldl min first)) from minimumFrom_encode rest first]
  rfl

end ProjectEulerSolutions.IntegerListMinimumCalls
