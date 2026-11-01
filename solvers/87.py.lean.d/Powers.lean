import solvers.«87.py.lean.d».Sieve
import ProjectEulerSolutions.Lib.PrimePowerTriplesPowers
import ProjectEulerSolutions.Lib.HeapCollection

namespace solvers.«87».buildPowers
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

structure Lists where
  squares : List Nat
  cubes : List Nat
  fourths : List Nat

def extend (limit : Nat) (items : List Nat) (before : Lists) : Lists :=
  ⟨before.squares ++ powerList limit 2 items, before.cubes ++ powerList limit 3 items,
    before.fourths ++ powerList limit 4 items⟩

theorem extend_nil (limit : Nat) (before : Lists) : extend limit [] before = before := by
  cases before
  simp [extend, powerList]

theorem extend_cons (limit p : Nat) (rest : List Nat) (before : Lists) :
    extend limit (p :: rest) before = extend limit rest (extend limit [p] before) := by
  change extend limit ([p] ++ rest) before = _
  simp only [extend, powerList_append, List.append_assoc]

def working (base : Mutable.Heap) (lists : Lists) : Mutable.Heap :=
  base ++ [.list (encode lists.squares), .list (encode lists.cubes), .list (encode lists.fourths)]

structure Locals (env : HeapCalls.Env) (limit primeAddress first : Nat) : Prop where
  bound : env "limit" = some (number limit)
  primes : env "primes" = some (.ref primeAddress)
  squares : env "squares" = some (.ref first)
  cubes : env "cubes" = some (.ref (first + 1))
  fourths : env "fourths" = some (.ref (first + 2))

theorem Locals.set_other {env : HeapCalls.Env} {limit primeAddress first : Nat}
    (ready : Locals env limit primeAddress first) (name : String) (value : Mutable.Value)
    (notLimit : name ≠ "limit") (notPrimes : name ≠ "primes")
    (notSquares : name ≠ "squares") (notCubes : name ≠ "cubes") (notFourths : name ≠ "fourths") :
    Locals (setLocal env name value) limit primeAddress first := by
  rcases ready with ⟨hl, hp, hs, hc, hf⟩
  constructor <;> simp_all [setLocal, Ne.symm]

abbrev appendBody (valueName listName : String) : List Stmt :=
  [.ifThen (.compare (.name valueName) [(.lt, .name "limit")])
    [.expr (.call (.attribute (.name listName) "append") [.name valueName])] []]

theorem conditional_append (limit value address : Nat) (items : List Nat) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (valueName listName : String)
    (getLimit : env "limit" = some (number limit)) (getValue : env valueName = some (number value))
    (getList : env listName = some (.ref address)) (found : heap[address]? = some (.list (encode items))) :
    BlockContinues program env (appendBody valueName listName) heap env
      (heap.set address (.list (encode (items ++ if value < limit then [value] else [])))) := by
  have tested : Evaluates program env (.compare (.name valueName) [(.lt, .name "limit")]) heap
      (.scalar (.bool (decide (value < limit)))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap)
      (Evaluates.lookup getValue) (Evaluates.lookup getLimit)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases inside : value < limit
  · apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockContinues.discard
    simpa [inside, encode, List.map_append] using
      (Evaluates.append (module := program) (globals := []) (Evaluates.lookup getList) found (Evaluates.lookup getValue) found)
  · have unchanged : heap.set address (.list (encode items)) = heap := by
      have entry := (List.getElem?_eq_some_iff.mp found).2
      rw [← entry]
      exact List.set_getElem_self (List.getElem?_eq_some_iff.mp found).1
    apply BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, inside])
    simpa [inside, unchanged] using (BlockContinues.nil (module := program) (globals := []) (env := env) (heap := heap))

abbrev body : List Stmt :=
  [.assign [.name "p2"] (.binOp (.name "p") .mult (.name "p"))] ++ appendBody "p2" "squares" ++
  [.assign [.name "p3"] (.binOp (.name "p2") .mult (.name "p"))] ++ appendBody "p3" "cubes" ++
  [.assign [.name "p4"] (.binOp (.name "p2") .mult (.name "p2"))] ++ appendBody "p4" "fourths"

theorem iteration (limit p : Nat) (base : Mutable.Heap) (primeAddress : Nat) (before : Lists)
    (env : HeapCalls.Env) (ready : Locals env limit primeAddress base.length)
    (getP : env "p" = some (number p)) :
    ∃ finalEnv, BlockContinues program env body (working base before) finalEnv
      (working base (extend limit [p] before)) ∧ Locals finalEnv limit primeAddress base.length := by
  let e2 := setLocal env "p2" (number (p ^ 2))
  let e3 := setLocal e2 "p3" (number (p ^ 3))
  let e4 := setLocal e3 "p4" (number (p ^ 4))
  let squareState : Lists := { before with squares := before.squares ++ powerList limit 2 [p] }
  let cubeState : Lists := { squareState with cubes := before.cubes ++ powerList limit 3 [p] }
  have ready2 := ready.set_other "p2" (number (p ^ 2)) (by decide) (by decide) (by decide) (by decide) (by decide)
  have ready3 := ready2.set_other "p3" (number (p ^ 3)) (by decide) (by decide) (by decide) (by decide) (by decide)
  have ready4 := ready3.set_other "p4" (number (p ^ 4)) (by decide) (by decide) (by decide) (by decide) (by decide)
  have first : BlockContinues program env [body[0]'(by decide)] (working base before) e2 (working base before) := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP) (by simp [Mutable.binary, scalarBinary, number, pow_two])
  have second : BlockContinues program e2 (appendBody "p2" "squares") (working base before) e2 (working base squareState) := by
    have executed := conditional_append limit (p ^ 2) base.length before.squares (working base before) e2 "p2" "squares"
      ready2.bound (by simp [e2, setLocal]) ready2.squares (by simp [working])
    simpa [working, squareState, powerList_singleton, List.set_append_right] using executed
  have third : BlockContinues program e2 [body[2]'(by decide)] (working base squareState) e3 (working base squareState) := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number (p ^ 2)) (bv := number p)
      (Evaluates.lookup (by simp [e2, setLocal])) (Evaluates.lookup (by simpa [e2, setLocal] using getP))
      (by simp [Mutable.binary, scalarBinary, number, pow_succ])
  have fourth : BlockContinues program e3 (appendBody "p3" "cubes") (working base squareState) e3 (working base cubeState) := by
    have executed := conditional_append limit (p ^ 3) (base.length + 1) before.cubes (working base squareState) e3 "p3" "cubes"
      ready3.bound (by simp [e3, setLocal]) ready3.cubes (by simp [working, squareState])
    simpa [working, cubeState, squareState, powerList_singleton, List.set_append_right] using executed
  have fifth : BlockContinues program e3 [body[4]'(by decide)] (working base cubeState) e4 (working base cubeState) := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number (p ^ 2)) (bv := number (p ^ 2))
      (Evaluates.lookup (by simp [e3, e2, setLocal])) (Evaluates.lookup (by simp [e3, e2, setLocal]))
    simp [Mutable.binary, scalarBinary, number, show p ^ 4 = p ^ 2 * p ^ 2 by ring]
  have sixth : BlockContinues program e4 (appendBody "p4" "fourths") (working base cubeState) e4
      (working base (extend limit [p] before)) := by
    have executed := conditional_append limit (p ^ 4) (base.length + 2) before.fourths (working base cubeState) e4 "p4" "fourths"
      ready4.bound (by simp [e4, setLocal]) ready4.fourths (by simp [working, cubeState, squareState])
    simpa [working, cubeState, squareState, powerList_singleton, extend, List.set_append_right] using executed
  exact ⟨e4, first.append (second.append (third.append (fourth.append (fifth.append sixth)))), ready4⟩

theorem loop (limit : Nat) (items rest : List Nat) (position : Nat) (suffix : items.drop position = rest)
    (base : Mutable.Heap) (primeAddress : Nat) (found : base[primeAddress]? = some (.list (encode items)))
    (before : Lists) (env : HeapCalls.Env) (ready : Locals env limit primeAddress base.length) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "p" (.ref primeAddress) position body []]
      (working base before) finalEnv (working base (extend limit rest before)) ∧
      Locals finalEnv limit primeAddress base.length := by
  have source (lists : Lists) : Mutable.sequence (.ref primeAddress) (working base lists) =
      some (.ok (encode items), working base lists) := by
    have kept := ProjectEulerSolutions.HeapCollection.prefix_found base
      [.list (encode lists.squares), .list (encode lists.cubes), .list (encode lists.fourths)] primeAddress _ found
    simp [Mutable.sequence, working, kept]
  induction rest generalizing position before env with
  | nil =>
      refine ⟨env, ?_, ready⟩
      rw [extend_nil]
      apply BlockContinues.sequence_done (source := .ref primeAddress) (items := encode items) (source before)
      · have missing : items[position]? = none := by simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [encode, missing]
      · exact .nil
  | cons head tail ih =>
      let bound := setLocal env "p" (number head)
      have boundReady := ready.set_other "p" (number head) (by decide) (by decide) (by decide) (by decide) (by decide)
      obtain ⟨nextEnv, iterated, nextReady⟩ := iteration limit head base primeAddress before bound boundReady (by simp [bound, setLocal])
      obtain ⟨finalEnv, continued, finalReady⟩ := ih (position + 1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) (extend limit [head] before) nextEnv nextReady
      refine ⟨finalEnv, ?_, finalReady⟩
      rw [extend_cons]
      have selected : items[position]? = some head := by simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      apply BlockContinues.sequence_step (source := .ref primeAddress) (items := encode items)
        (value := number head) (source before) (by simp [encode, selected])
      simpa only [Calls.loopContinuation, show Calls.loopControl body = false from rfl,
        Bool.false_eq_true, if_false] using iterated.append continued

theorem result (limit : Nat) (items : List Nat) (base : Mutable.Heap) (primeAddress : Nat)
    (found : base[primeAddress]? = some (.list (encode items))) (before : Lists)
    (env : HeapCalls.Env) (ready : Locals env limit primeAddress base.length) :
    ∃ finalEnv, BlockContinues program env [.forIn (.name "p") (.name "primes") body []]
      (working base before) finalEnv (working base (extend limit items before)) ∧
      Locals finalEnv limit primeAddress base.length := by
  obtain ⟨finalEnv, executed, finalReady⟩ := loop limit items items 0 rfl base primeAddress found before env ready
  refine ⟨finalEnv, ?_, finalReady⟩
  apply BlockContinues.for_sequence (value := .ref primeAddress) (items := encode items)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup ready.primes)
  · have kept := ProjectEulerSolutions.HeapCollection.prefix_found base
      [.list (encode before.squares), .list (encode before.cubes), .list (encode before.fourths)] primeAddress _ found
    simp [Mutable.sequence, working, kept]
  · exact executed

end solvers.«87».buildPowers
