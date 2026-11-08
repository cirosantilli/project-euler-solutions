import solvers.«84.py.lean.d».LandingPool

namespace solvers.«84».resolve_landing
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

def SingletonPost (base : Mutable.Heap) (square : Int) (jailed : Bool) (weight : Int)
    (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ keyAddress address extra, value = .ref address ∧ after = base ++ extra ∧ base.length ≤ address ∧
    after[address]? = some (.keyDict [(.tuple (pool.pair square jailed), .ref keyAddress, number weight)]) ∧
    after[keyAddress]? = some (.tuple [number square, .scalar (.bool jailed)])

/-- A dictionary display preserves the actual cached key reference and allocates a fresh dictionary. -/
theorem singleton_pooled (expression : Expr) (square : Int) (jailed : Bool) (weight : Int)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (key : HeapMemoCalls.EvaluatesRel program pool.spec env expression heap
      (TuplePool.Post heap (.tuple (pool.pair square jailed))) globals) :
    HeapMemoCalls.EvaluatesRel program pool.spec env (.dict [(expression, .constant (.int weight))]) heap
      (SingletonPost heap square jailed weight) globals := by
  intro anchors stack
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.dictionary [(expression, .constant (.int weight))] [] env, stack, heap⟩) (by rfl) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.expr expression env, .dictionaryKey (.constant (.int weight)) [] [] env :: stack, heap⟩) (by trivial) rfl
  apply (key anchors _).bind
  rintro final ⟨value, middle, rfl, extra, same, view⟩
  obtain ⟨address, rfl, found⟩ := (TuplePool.view_scalars middle [.int square, .bool jailed] value).mp view
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.expr (.constant (.int weight)) env,
      .arguments .dictionarySingleton [.ref address] [] env :: stack, middle⟩) (by trivial) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.value (number weight), .arguments .dictionarySingleton [.ref address] [] env :: stack, middle⟩) (by rfl) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.operation .dictionarySingleton [.ref address, number weight], stack, middle⟩) (by trivial) rfl
  let dictionary := Mutable.Object.keyDict [(.tuple (pool.pair square jailed), .ref address, number weight)]
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.value (.ref middle.length), stack, middle ++ [dictionary]⟩)
    (preserved := TuplePool.preserves_append pool.catalog _ _) (by trivial)
  · simp [HeapCalls.step, HeapCalls.primitive, HeapCalls.result, HeapCalls.run,
      DictionaryDisplay.singleton_result middle (.ref address) (number weight) _
        (Mutable.hashKey_tuple_scalars middle address [.int square, .bool jailed] found), dictionary, pool.pair]
  · apply HeapMemoCalls.RunsRel.refl
    refine ⟨.ref middle.length, middle ++ [dictionary], rfl, address, middle.length,
      extra ++ [dictionary], rfl, by simp [same, List.append_assoc], ?_, by simp [dictionary], ?_⟩
    · simp [same]
    · exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found

/-- G2J always sends the player to jail and marks that the doubles run must reset. -/
theorem jail_result (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program pool.spec "resolve_landing" [number 30] heap
      (SingletonPost heap 10 true 256) globals := by
  let env := HeapCalls.functionEnv pool.definition.1 pool.definition.2 [number 30] globals
  have getSquare : env "square" = some (number 30) := by cases globals <;> rfl
  have tested : HeapMemoCalls.Evaluates program pool.spec env (.compare (.name "square") [(.eq, .constant (.int 30))])
      heap (.scalar (.bool true)) heap globals :=
    HeapMemoCalls.Evaluates.compare (HeapMemoCalls.Evaluates.lookup getSquare) HeapMemoCalls.Evaluates.constant rfl
  apply HeapMemoCalls.BlockReturnsRel.function (args := pool.definition.1) (body := pool.definition.2) (by decide) rfl rfl
  apply HeapMemoCalls.BlockReturnsRel.branch tested.rel
  rintro value after ⟨rfl, same⟩
  subst after
  refine ⟨true, rfl, ?_⟩
  apply HeapMemoCalls.BlockReturnsRel.return_value
  exact singleton_pooled pool.jailKey 10 true 256 env heap globals
    (pool.pair_result pool.jailKey 10 true pool.jail_source pool.known_jail env heap globals)

end solvers.«84».resolve_landing
