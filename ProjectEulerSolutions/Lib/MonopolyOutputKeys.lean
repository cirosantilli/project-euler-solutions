import ProjectEulerSolutions.Lib.MonopolyOutputSorting

namespace ProjectEulerSolutions.MonopolyOutput
open FormalSW.PyAst HeapCalls
set_option Elab.async false

/-- Key callbacks allocate one tuple each, and cached keys remain readable until sorting finishes. -/
theorem keys_result (module : Module) (values : Fin 120 → Int) (remaining : List (Fin 40))
    (keys : List (Fin 40 × Nat)) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode (Monopoly.squareWeight values))))
    (cached : ∀ entry ∈ keys, heap[entry.2]? = some (keyObject values entry.1))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (parameter : String) (different : parameter ≠ "square_weights")
    (getWeights : env "square_weights" = some (.ref address)) :
    ∃ extra result, SortedReturns module ⟨parameter, keyExpression parameter, env, false⟩
      (remaining.map fun square => .scalar (.int square.val)) (keys.map cachedPair) heap (.ref result) (heap ++ extra) globals ∧
      (heap ++ extra)[result]? = some (.list
        ((HeapSort.sortBy (less values) ((keys.map Prod.fst).reverse ++ remaining)).map
          fun square => .scalar (.int square.val))) := by
  induction remaining generalizing keys heap with
  | nil =>
      refine ⟨[.list ((HeapSort.sortBy (less values) (keys.map Prod.fst).reverse).map
        fun square => .scalar (.int square.val))], heap.length, ?_, by simp⟩
      exact SortedReturns.done (finish_keys values heap keys cached)
  | cons square rest ih =>
      have evaluated := key_result module values square heap address found
        (setLocal env parameter (.scalar (.int square.val))) globals parameter
        (by simpa [setLocal, Ne.symm different] using getWeights) (by simp [setLocal])
      have nextCached : ∀ entry ∈ (square, heap.length) :: keys,
          (heap ++ [keyObject values square])[entry.2]? = some (keyObject values entry.1) := by
        intro entry member
        rcases List.mem_cons.mp member with rfl | old
        · simp
        · exact HeapCollection.prefix_found heap _ entry.2 _ (cached entry old)
      obtain ⟨extra, result, ran, final⟩ := ih ((square, heap.length) :: keys) (heap ++ [keyObject values square])
        (HeapCollection.prefix_found heap _ address _ found) nextCached
      refine ⟨[keyObject values square] ++ extra, result, ?_, ?_⟩
      · simpa only [List.map_cons, cachedPair, List.append_assoc] using SortedReturns.next evaluated ran
      · simpa only [List.map_cons, List.reverse_cons, List.append_assoc, List.singleton_append] using final

def sortedSquares (values : Fin 120 → Int) : List (Fin 40) :=
  HeapSort.sortBy (less values) (List.finRange 40)

def squareIterable : Expr := .listComp (.name "$formalsw.range.item")
  [("$formalsw.range.item", .call (.name "range") [.constant (.int 40)], [])]

def sortExpression (parameter : String) : Expr :=
  .sortedSequence squareIterable parameter (keyExpression parameter) (.constant (.bool false))

theorem squares_encoding : (List.range' 0 40).map (fun number : Nat => Mutable.Value.scalar (.int number)) =
    (List.finRange 40).map (fun square => .scalar (.int square.val)) := by
  apply List.ext_getElem
  · simp only [List.length_map, List.length_range', List.length_finRange]
  · intro index left right
    simp only [List.getElem_map, List.getElem_range', Nat.zero_add, Nat.one_mul, List.getElem_finRange, Fin.val_cast]

/-- The range snapshot, key tuples and sorted list are private allocations in the actual source evaluation. -/
theorem sort_result (module : Module) (values : Fin 120 → Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode (Monopoly.squareWeight values))))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (parameter : String) (different : parameter ≠ "square_weights")
    (getWeights : env "square_weights" = some (.ref address)) :
    ∃ extra result, Evaluates module env (sortExpression parameter) heap (.ref result) (heap ++ extra) globals ∧
      (heap ++ extra)[result]? = some (.list ((sortedSquares values).map fun square => .scalar (.int square.val))) := by
  obtain ⟨garbage, iterable⟩ := HeapCollection.range_map (globals := globals) module "$formalsw.range.item" (.name "$formalsw.range.item")
    (fun number => .scalar (.int number))
    (fun _ _ _ looked => ⟨[], by simpa using (Evaluates.lookup looked)⟩)
    heap env [.constant (.int 40)] [.scalar (.int 40)] 0 40 (.cons Evaluates.constant .nil) rfl
  rw [squares_encoding] at iterable
  let middle := HeapCollection.working heap ((List.finRange 40).map fun square => .scalar (.int square.val)) garbage
  obtain ⟨extra, result, ran, final⟩ := keys_result module values (List.finRange 40) [] middle address
    (HeapCollection.prefix_found heap _ address _ found) (by simp) env globals parameter different getWeights
  refine ⟨(.list ((List.finRange 40).map fun square => .scalar (.int square.val)) :: garbage) ++ extra, result, ?_, ?_⟩
  · have acquired : sortedStart (.ref heap.length) (.scalar (.bool false)) middle = some
        (.ok ((List.finRange 40).map (fun square => .scalar (.int square.val)), false), middle) := by
      simp [sortedStart, sortedItems, Mutable.sequence, middle, Mutable.truth, scalarTruth]
    have evaluated := Evaluates.sorted_sequence iterable Evaluates.constant acquired ran
    simpa only [sortExpression, squareIterable, middle, HeapCollection.working, List.append_assoc] using evaluated
  · simpa only [sortedSquares, List.map_nil, List.reverse_nil, List.nil_append, middle,
      HeapCollection.working, List.append_assoc] using final

end ProjectEulerSolutions.MonopolyOutput
