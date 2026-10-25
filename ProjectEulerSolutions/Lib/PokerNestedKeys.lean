import lang.py.FormalSW.TupleComparisons
import Lean.Elab.Tactic.Omega

namespace ProjectEulerSolutions.PokerNestedKeys
open FormalSW.PyAst
set_option Elab.async false

private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number

private theorem lexInt_self (values : List Int) : HeapSort.lexInt values values = false := by
  induction values with
  | nil => rfl
  | cons head tail ih => simpa only [HeapSort.lexInt, if_pos rfl] using ih

private theorem numbers_injective (left right : List Int) (same : numbers left = numbers right) : left = right := by
  have decoded := congrArg (List.filterMap fun value => match value with
    | .scalar (.int n) => some n
    | _ => none) same
  simpa [numbers, List.filterMap_map, Function.comp_def] using decoded

private theorem lex_ints_equal (heap : Mutable.Heap) (depth : Nat) (left right : List Int) :
    HeapSort.lex (HeapSort.compare (depth+1) heap true) (HeapSort.compare (depth+1) heap false) true
      (numbers left) (numbers right) = some (.ok (decide (left = right))) := by
  induction left generalizing right with
  | nil => cases right <;> rfl
  | cons head tail ih =>
      cases right with
      | nil => rfl
      | cons other rest =>
          have equal : HeapSort.compare (depth+1) heap true (number head) (number other) =
              some (.ok (decide (head = other))) := rfl
          simp only [numbers, List.map_cons, HeapSort.lex, equal]
          by_cases same : head = other
          · subst other
            simpa using ih rest
          · simp [same]

/-- Equality of actual integer tuples is total at every depth with room for their scalar elements. -/
theorem compare_int_tuples_equal (heap : Mutable.Heap) (leftAddress rightAddress depth : Nat)
    (left right : List Int)
    (leftFound : heap[leftAddress]? = some (.tuple (numbers left)))
    (rightFound : heap[rightAddress]? = some (.tuple (numbers right))) :
    HeapSort.compare (depth+2) heap true (.ref leftAddress) (.ref rightAddress) =
      some (.ok (decide (left = right))) := by
  by_cases alias : leftAddress = rightAddress
  · subst rightAddress
    have contents : left = right := numbers_injective left right (by simpa using leftFound.symm.trans rightFound)
    subst right
    simp [HeapSort.compare, leftFound]
  · by_cases lengths : left.length = right.length
    · rw [HeapSort.compare, leftFound, rightFound]
      simp [alias, numbers, lengths, lex_ints_equal]
    · have different : left ≠ right := fun same => lengths (congrArg List.length same)
      simp [HeapSort.compare, leftFound, rightFound, alias, numbers, lengths, different]

/-- Equal-size scalar prefixes followed by one integer-tuple field compare exactly like flattened integer keys. -/
theorem lex_nested (heap : Mutable.Heap) (leftTailAddress rightTailAddress depth : Nat)
    (leftTail rightTail leftPrefix rightPrefix : List Int)
    (leftFound : heap[leftTailAddress]? = some (.tuple (numbers leftTail)))
    (rightFound : heap[rightTailAddress]? = some (.tuple (numbers rightTail)))
    (lengths : leftPrefix.length = rightPrefix.length) :
    HeapSort.lex (HeapSort.compare (depth+2) heap true) (HeapSort.compare (depth+2) heap false) false
      (numbers leftPrefix ++ [.ref leftTailAddress]) (numbers rightPrefix ++ [.ref rightTailAddress]) =
        some (.ok (HeapSort.lexInt (leftPrefix ++ leftTail) (rightPrefix ++ rightTail))) := by
  induction leftPrefix generalizing rightPrefix with
  | nil =>
      cases rightPrefix with
      | cons head tail => simp at lengths
      | nil =>
          have equal := compare_int_tuples_equal heap leftTailAddress rightTailAddress depth leftTail rightTail leftFound rightFound
          have less := HeapSort.compare_int_tuples heap leftTailAddress rightTailAddress depth leftTail rightTail leftFound rightFound
          change HeapSort.lex _ _ false [.ref leftTailAddress] [.ref rightTailAddress] = some (.ok (HeapSort.lexInt leftTail rightTail))
          rw [HeapSort.lex, equal]
          by_cases same : leftTail = rightTail
          · subst rightTail
            simp [HeapSort.lex, lexInt_self]
          · simp [same, less]
  | cons head tail ih =>
      cases rightPrefix with
      | nil => simp at lengths
      | cons other rest =>
          have restLengths : tail.length = rest.length := by simpa only [List.length_cons, Nat.succ.injEq] using lengths
          have equal : HeapSort.compare (depth+2) heap true (number head) (number other) =
              some (.ok (decide (head = other))) := rfl
          have less : HeapSort.compare (depth+2) heap false (number head) (number other) =
              some (.ok (decide (head < other))) := rfl
          simp only [numbers, List.map_cons, List.cons_append, HeapSort.lex, equal]
          by_cases same : head = other
          · subst other
            simpa only [decide_true, HeapSort.lexInt, if_pos rfl] using ih rest restLengths
          · simp [same, less, HeapSort.lexInt]

theorem compare_nested (heap : Mutable.Heap) (leftAddress rightAddress leftTailAddress rightTailAddress depth : Nat)
    (leftPrefix rightPrefix leftTail rightTail : List Int)
    (leftFound : heap[leftAddress]? = some (.tuple (numbers leftPrefix ++ [.ref leftTailAddress])))
    (rightFound : heap[rightAddress]? = some (.tuple (numbers rightPrefix ++ [.ref rightTailAddress])))
    (leftTailFound : heap[leftTailAddress]? = some (.tuple (numbers leftTail)))
    (rightTailFound : heap[rightTailAddress]? = some (.tuple (numbers rightTail)))
    (lengths : leftPrefix.length = rightPrefix.length) :
    HeapSort.compare (depth+3) heap false (.ref leftAddress) (.ref rightAddress) =
      some (.ok (HeapSort.lexInt (leftPrefix ++ leftTail) (rightPrefix ++ rightTail))) := by
  simpa only [HeapSort.compare, leftFound, rightFound, Bool.false_and, Bool.false_eq_true, ↓reduceIte] using
    lex_nested heap leftTailAddress rightTailAddress depth leftTail rightTail leftPrefix rightPrefix leftTailFound rightTailFound lengths

private theorem nested_room (heap : Mutable.Heap) (outerAddress tailAddress : Nat) (initial tail : List Int)
    (outerFound : heap[outerAddress]? = some (.tuple (numbers initial ++ [.ref tailAddress])))
    (tailFound : heap[tailAddress]? = some (.tuple (numbers tail))) : 2 ≤ heap.length := by
  have outerInside := (List.getElem?_eq_some_iff.mp outerFound).1
  have tailInside := (List.getElem?_eq_some_iff.mp tailFound).1
  have distinct : outerAddress ≠ tailAddress := by
    intro same
    subst outerAddress
    have contents : numbers initial ++ [.ref tailAddress] = numbers tail := by
      simpa only [Option.some.injEq, Mutable.Object.tuple.injEq] using outerFound.symm.trans tailFound
    have member : Mutable.Value.ref tailAddress ∈ numbers initial ++ [.ref tailAddress] := by simp
    rw [contents] at member
    simp [numbers] at member
  omega

/-- Real Python greater-than on the nested final poker tuples is flattened lexicographic order.
The heap lookups themselves supply enough recursive depth; no heap-validity or size premise is needed. -/
theorem compare_gt (heap : Mutable.Heap) (leftAddress rightAddress leftTailAddress rightTailAddress : Nat)
    (leftPrefix rightPrefix leftTail rightTail : List Int)
    (leftFound : heap[leftAddress]? = some (.tuple (numbers leftPrefix ++ [.ref leftTailAddress])))
    (rightFound : heap[rightAddress]? = some (.tuple (numbers rightPrefix ++ [.ref rightTailAddress])))
    (leftTailFound : heap[leftTailAddress]? = some (.tuple (numbers leftTail)))
    (rightTailFound : heap[rightTailAddress]? = some (.tuple (numbers rightTail)))
    (lengths : leftPrefix.length = rightPrefix.length) :
    HeapCalls.compare .gt (.ref leftAddress) (.ref rightAddress) heap =
      some (.ok (.scalar (.bool (HeapSort.lexInt (rightPrefix ++ rightTail) (leftPrefix ++ leftTail)))), heap) := by
  have room := nested_room heap leftAddress leftTailAddress leftPrefix leftTail leftFound leftTailFound
  have depth : heap.length-2+3 = heap.length+1 := by omega
  apply HeapCalls.compare_tuple_gt heap leftAddress rightAddress _ _ _ leftFound rightFound
  simpa only [depth] using compare_nested heap rightAddress leftAddress rightTailAddress leftTailAddress (heap.length-2)
    rightPrefix leftPrefix rightTail leftTail rightFound leftFound rightTailFound leftTailFound lengths.symm

end ProjectEulerSolutions.PokerNestedKeys
