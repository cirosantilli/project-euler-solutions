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

theorem minimum_primitive (heap : Mutable.Heap) (address : Nat) (first : Int) (rest : List Int)
    (found : heap[address]? = some (.list (encode (first :: rest)))) :
    primitive (.builtin .min) [.ref address] heap = some (.ok (number (rest.foldl min first)), heap) := by
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

theorem minimum_expression (program : Module) (env : HeapCalls.Env) (expression : Expr)
    (heap middle : Mutable.Heap) (address : Nat) (first : Int) (rest : List Int)
    (evaluated : Evaluates program env expression heap (.ref address) middle)
    (found : middle[address]? = some (.list (encode (first :: rest)))) :
    Evaluates program env (.builtinCall .min [expression]) heap (number (rest.foldl min first)) middle :=
  Evaluates.builtin (.cons evaluated .nil) (minimum_primitive middle address first rest found)

theorem minimum_evaluate (program : Module) (name : String) (first : Int) (rest : List Int) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (address : Nat) (localValue : env name = some (.ref address))
    (found : heap[address]? = some (.list (encode (first :: rest)))) :
    Evaluates program env (.builtinCall .min [.name name]) heap
      (number (rest.foldl min first)) heap :=
  minimum_expression program env (.name name) heap heap address first rest (Evaluates.lookup localValue) found

theorem fold_min_insert (values : List Int) (initial candidate : Int) :
    values.foldl min (min initial candidate) = min (values.foldl min initial) candidate := by
  induction values generalizing initial with
  | nil => rfl
  | cons first rest ih =>
      simp only [List.foldl_cons]
      rw [show min (min initial candidate) first = min (min initial first) candidate by omega, ih]

theorem minimum_with_last (program : Module) (env : HeapCalls.Env) (expression : Expr)
    (heap middle : Mutable.Heap) (address : Nat) (values : List Int) (last : Int)
    (evaluated : Evaluates program env expression heap (.ref address) middle)
    (found : middle[address]? = some (.list (encode (values ++ [last])))) :
    Evaluates program env (.builtinCall .min [expression]) heap (number (values.foldl min last)) middle := by
  cases values with
  | nil => exact minimum_expression program env expression heap middle address last [] evaluated found
  | cons first rest =>
      have result := minimum_expression program env expression heap middle address first (rest ++ [last]) evaluated found
      rw [List.foldl_append, List.foldl_cons, List.foldl_nil, ← fold_min_insert, min_comm first last] at result
      exact result

theorem minimum_pair (program : Module) (env : HeapCalls.Env) (left right : Expr)
    (heap middle finalHeap : Mutable.Heap) (x y : Int)
    (first : Evaluates program env left heap (number x) middle)
    (second : Evaluates program env right middle (number y) finalHeap) :
    Evaluates program env (.builtinCall .min [left, right]) heap (number (min x y)) finalHeap := by
  apply Evaluates.builtin (.cons first (.cons second .nil))
  by_cases smaller : y < x
  · simp [primitive, scalarResult, Calls.positionalMinimum, Calls.minimumFrom, scalarCompare, evalCmpOp, smaller,
      Int.min_eq_right (by omega : y ≤ x), List.mapM, List.mapM.loop]
  · simp [primitive, scalarResult, Calls.positionalMinimum, Calls.minimumFrom, scalarCompare, evalCmpOp, smaller,
      Int.min_eq_left (by omega : x ≤ y), List.mapM, List.mapM.loop]

end ProjectEulerSolutions.IntegerListMinimumCalls
