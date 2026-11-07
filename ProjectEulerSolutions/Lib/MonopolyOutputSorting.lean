import ProjectEulerSolutions.Lib.MonopolyOutputCollection
import ProjectEulerSolutions.Lib.HeapKeySorting

namespace ProjectEulerSolutions.MonopolyOutput
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def keyNumbers (values : Fin 120 → Int) (square : Fin 40) : List Int :=
  [-Monopoly.squareWeight values square, square.val]

def keyObject (values : Fin 120 → Int) (square : Fin 40) : Mutable.Object :=
  .tuple ((keyNumbers values square).map fun number => .scalar (.int number))

def less (values : Fin 120 → Int) (first second : Fin 40) : Bool :=
  HeapSort.lexInt (keyNumbers values first) (keyNumbers values second)

theorem less_iff (values : Fin 120 → Int) (first second : Fin 40) :
    less values first second = true ↔ Monopoly.ranksBefore values first second := by
  by_cases equal : Monopoly.squareWeight values first = Monopoly.squareWeight values second
  · by_cases same : first.val = second.val
    · simp [less, keyNumbers, HeapSort.lexInt, Monopoly.ranksBefore, equal, same]
    · simp [less, keyNumbers, HeapSort.lexInt, Monopoly.ranksBefore, equal, same]
  · simp [less, keyNumbers, HeapSort.lexInt, Monopoly.ranksBefore, equal]

/-- Exact normalization cannot change the sorted board order, including tied weights. -/
theorem sort_equal {sides : Nat} (positive : 0 < sides) {left right : Fin 120 → Int}
    (validLeft : Monopoly.IntegerWeights sides left) (validRight : Monopoly.IntegerWeights sides right)
    (squares : List (Fin 40)) :
    HeapSort.sortBy (less left) squares = HeapSort.sortBy (less right) squares := by
  have same : less left = less right := by
    funext first second
    apply Bool.eq_iff_iff.mpr
    rw [less_iff, less_iff]
    exact validLeft.ranking_iff positive validRight first second
  rw [same]

def keyExpression (parameter : String) : Expr := .tuple
  [.unaryOp .uSub (.subscript (.name "square_weights") (.name parameter)), .name parameter]

theorem key_result (module : Module) (values : Fin 120 → Int) (square : Fin 40)
    (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode (Monopoly.squareWeight values))))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (parameter : String)
    (getWeights : env "square_weights" = some (.ref address))
    (getSquare : env parameter = some (.scalar (.int square.val))) :
    Evaluates module env (keyExpression parameter) heap (.ref heap.length)
      (heap ++ [keyObject values square]) globals := by
  have read : Evaluates module env (.subscript (.name "square_weights") (.name parameter)) heap
      (.scalar (.int (Monopoly.squareWeight values square))) heap globals :=
    Evaluates.subscript (by intros; simp) (.lookup getWeights) (.lookup getSquare)
      (by simpa only [encode, List.getElem_ofFn, Fin.eta] using
            Mutable.getItem_list heap address square.val _ found
              (by simpa only [encode, List.length_ofFn] using square.isLt))
  exact Evaluates.tuple (.cons (Evaluates.unary read rfl) (.cons (.lookup getSquare) .nil)) (by simp [keyNumbers])

def cachedPair (entry : Fin 40 × Nat) : Mutable.Value × Mutable.Value :=
  (.scalar (.int entry.1.val), .ref entry.2)

theorem compared_keys (values : Fin 120 → Int) (heap : Mutable.Heap) (first second : Fin 40 × Nat)
    (left : heap[first.2]? = some (keyObject values first.1))
    (right : heap[second.2]? = some (keyObject values second.1)) :
    HeapSort.compare (HeapSort.comparisonFuel heap) heap false (cachedPair first).2 (cachedPair second).2 =
      some (.ok (less values first.1 second.1)) := by
  have inside := (List.getElem?_eq_some_iff.mp left).1
  have allowance := HeapSort.comparisonFuel_ge_linear heap
  have depth : HeapSort.comparisonFuel heap - 2 + 2 = HeapSort.comparisonFuel heap := by omega
  simpa only [depth, cachedPair, less] using
    HeapSort.compare_int_tuples heap first.2 second.2 (HeapSort.comparisonFuel heap - 2)
      (keyNumbers values first.1) (keyNumbers values second.1) left right

/-- Once all callbacks have run, sorting reads the saved tuple keys without further effects. -/
theorem finish_keys (values : Fin 120 → Int) (heap : Mutable.Heap) (keys : List (Fin 40 × Nat))
    (found : ∀ entry ∈ keys, heap[entry.2]? = some (keyObject values entry.1)) :
    sortedFinish false (keys.map cachedPair) heap = some (.ok (.ref heap.length),
      heap ++ [.list ((HeapSort.sortBy (less values) (keys.map Prod.fst).reverse).map
        fun square => .scalar (.int square.val))]) := by
  have sorted := HeapKeySorting.sort_map heap cachedPair
    (fun first second => less values first.1 second.1) keys.reverse
    (fun first hf second hs => compared_keys values heap first second
      (found first (List.mem_reverse.mp hf)) (found second (List.mem_reverse.mp hs)))
  have projected := HeapSort.sortBy_map (Prod.fst : Fin 40 × Nat → Fin 40) (less values) keys.reverse
  have payload : ((HeapSort.sortBy (fun first second => less values first.1 second.1) keys.reverse).map
      cachedPair).map Prod.fst = (HeapSort.sortBy (less values) (keys.map Prod.fst).reverse).map
        (fun square => .scalar (.int square.val)) := by
    simpa only [List.map_map, List.map_reverse, cachedPair] using
      congrArg (List.map fun square : Fin 40 => Mutable.Value.scalar (.int square.val)) projected
  simp [sortedFinish, HeapSort.keyedSortWithReverse, ← List.map_reverse, sorted, payload, Mutable.allocate, Except.map]

end ProjectEulerSolutions.MonopolyOutput
