import solvers.«93.py.lean.d».PairFinish

namespace solvers.«93».pair_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

/-- The optimized arithmetic helper accepts shared input tuples and preserves the entire caller heap. -/
theorem result (a b : Pair) (left right : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (positiveA : 0 < a.2) (positiveB : 0 < b.2)
    (foundA : heap[left]? = some (object a)) (foundB : heap[right]? = some (object b)) :
    ∃ extra, ReturnsValue program "pair_results" [.ref left, .ref right] heap (.ref (listAddress heap))
      (heap ++ extra) globals ∧ Heap.ListFound (heap ++ extra) (listAddress heap) (choices a b) := by
  let args := ((findFunctionIn program.body "pair_results").getD ([], [])).1
  let source := ((findFunctionIn program.body "pair_results").getD ([], [])).2
  let entry := functionEnv args source [.ref left, .ref right] globals
  have getA : entry "a" = some (.ref left) := by cases globals <;> rfl
  have getB : entry "b" = some (.ref right) := by cases globals <;> rfl
  let first := assignNames entry ["numerator_a", "denominator_a"] [number a.1, number a.2]
  let second := assignNames first ["numerator_b", "denominator_b"] [number b.1, number b.2]
  let multiplied := setLocal second "denominator" (number (a.2 * b.2))
  have ready : Ready multiplied a b := by
    constructor <;> simp [multiplied, second, first, assignNames, setLocal]
  let complete := setLocal multiplied "results" (.ref (listAddress heap))
  have readyOut : Ready complete a b := by
    constructor <;> simp [complete, multiplied, second, first, assignNames, setLocal]
  obtain ⟨extra, finished, finalFound⟩ := finish_result heap a b positiveA positiveB complete globals readyOut
    (by simp [complete, setLocal])
  refine ⟨extra, ?_, finalFound⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.unpack (names := ["numerator_a", "denominator_a"])
    (items := [number a.1, number a.2]) (afterUnpack := heap) rfl (Evaluates.lookup getA)
    (by simp [Mutable.sequence, foundA, object, Heap.object]) rfl
  apply BlockReturns.unpack (names := ["numerator_b", "denominator_b"])
    (items := [number b.1, number b.2]) (afterUnpack := heap) rfl
    (Evaluates.lookup (by simpa [first, assignNames, setLocal] using getB))
    (by simp [Mutable.sequence, foundB, object, Heap.object]) rfl
  apply BlockReturns.assign
    (times (x := a.2) (y := b.2) (Evaluates.lookup (by simp [assignNames, setLocal]))
      (Evaluates.lookup (by simp [assignNames, setLocal])))
  apply BlockReturns.assign (initial a b positiveA positiveB heap multiplied globals ready)
  exact finished

def Verified (a b : Pair) (left right : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ∃ extra, ReturnsValue program "pair_results" [.ref left, .ref right] heap (.ref (listAddress heap))
    (heap ++ extra) globals ∧ Heap.ListFound (heap ++ extra) (listAddress heap) (choices a b)

/-- Every returned key is canonical and the represented values cover both ordered operation choices. -/
theorem values (a b : Pair) (left right : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (positiveA : 0 < a.2) (positiveB : 0 < b.2)
    (foundA : heap[left]? = some (object a)) (foundB : heap[right]? = some (object b)) :
    ∃ extra pairs, ReturnsValue program "pair_results" [.ref left, .ref right] heap (.ref (listAddress heap))
      (heap ++ extra) globals ∧ Heap.ListFound (heap ++ extra) (listAddress heap) pairs ∧
      (∀ pair ∈ pairs, Canonical pair ∧ 0 < pair.2) ∧
      (∀ q : Rat, q ∈ pairs.map value ↔
        q ∈ ProjectEulerSolutions.ArithmeticExpressions.operations (value a) (value b) ∨
        q ∈ ProjectEulerSolutions.ArithmeticExpressions.operations (value b) (value a)) := by
  obtain ⟨extra, executed, found⟩ := result a b left right heap globals positiveA positiveB foundA foundB
  refine ⟨extra, choices a b, executed, found, ?_, mem_choices_values a b positiveA positiveB⟩
  intro pair member
  have canonical := choices_canonical a b positiveA positiveB pair member
  exact ⟨canonical, canonical.positive⟩

end solvers.«93».pair_results
