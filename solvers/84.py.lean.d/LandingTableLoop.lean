import solvers.«84.py.lean.d».LandingTableView

namespace solvers.«84».landings
open FormalSW.PyAst ProjectEulerSolutions resolve_landing
set_option Elab.async false

def element : Expr := .call (.name "resolve_landing") [.name "pos"]

theorem element_result (square : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getSquare : env "pos" = some (.scalar (.int square))) :
    HeapMemoCalls.EvaluatesRel program pool.spec env element heap
      (fun value after => ∃ (dictionary : Dictionary) (extra : Mutable.Heap), value = .ref dictionary.address ∧ after = heap ++ extra ∧ dictionary.View after square) globals := by
  apply HeapMemoCalls.EvaluatesRel.call
    (HeapMemoCalls.ArgumentsRel.fixed (.cons (HeapMemoCalls.Evaluates.lookup getSquare) .nil))
  rintro values after ⟨rfl, same⟩
  subst after
  exact dictionary_result square heap globals

theorem loop (count first stop : Nat) (length : first + count = stop)
    (base garbage : Mutable.Heap) (completed : List Nat) (dictionaries : List Dictionary)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (views : Views (HeapCollection.working base (references dictionaries) garbage) completed dictionaries)
    (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : ∀ dictionaries garbage,
      Views (HeapCollection.working base (references dictionaries) garbage)
        (completed ++ List.range' first count) dictionaries →
      post (.scalar .none) (HeapCollection.working base (references dictionaries) garbage)) :
    HeapMemoCalls.BlockReturnsRel program pool.spec env
      [.rangeStepLoop "pos" first stop 1 [.comprehensionAppend base.length element] []]
      (HeapCollection.working base (references dictionaries) garbage) post globals := by
  induction count generalizing first garbage completed dictionaries env with
  | zero =>
      apply HeapMemoCalls.BlockReturnsRel.range_done (by simp [Calls.rangeContinues]; omega)
      apply MemoCollection.nil
      exact rest dictionaries garbage (by simpa using views)
  | succ count ih =>
      apply HeapMemoCalls.BlockReturnsRel.range_step (by simp [Calls.rangeContinues]; omega)
      change HeapMemoCalls.BlockReturnsRel program pool.spec
        (HeapCalls.setLocal env "pos" (.scalar (.int first)))
        [.comprehensionAppend base.length element,
          .rangeStepLoop "pos" ((first : Int) + 1) stop 1 [.comprehensionAppend base.length element] []]
        (HeapCollection.working base (references dictionaries) garbage) post globals
      apply MemoCollection.emit (element_result first _ _ globals (by simp [HeapCalls.setLocal]))
      rintro value after ⟨dictionary, extra, rfl, rfl, view⟩
      have found := HeapCollection.prefix_found _ extra _ _ (HeapCollection.found base (references dictionaries) garbage)
      refine ⟨references dictionaries, found,
        TuplePool.preserves_frame pool.catalog _ _ (MemoCollection.tuple_frame _ _ _ _ found), ?_⟩
      have allViews := (views.append extra).snoc first dictionary view
      have stored := allViews.store base.length (references dictionaries) (references dictionaries ++ [.ref dictionary.address]) found
      have layout : (HeapCollection.working base (references dictionaries) garbage ++ extra).set base.length
          (.list (references dictionaries ++ [.ref dictionary.address])) =
          HeapCollection.working base (references (dictionaries ++ [dictionary])) (garbage ++ extra) := by
        rw [HeapCollection.append, HeapCollection.store]
        simp [references]
      rw [layout] at stored ⊢
      have continued := ih (first + 1) (by omega) (garbage ++ extra) (completed ++ [first])
        (dictionaries ++ [dictionary]) (HeapCalls.setLocal env "pos" (.scalar (.int first))) stored
      simp only [Int.natCast_add, Int.natCast_one] at continued
      apply continued
      intro finalDictionaries finalGarbage finalViews
      apply rest finalDictionaries finalGarbage
      simpa only [List.range'_succ, List.append_assoc, List.singleton_append] using finalViews

end solvers.«84».landings
