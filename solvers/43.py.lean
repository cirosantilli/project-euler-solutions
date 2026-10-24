import «.formalsw».solvers.«43.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p43.py»
import ProjectEulerSolutions.Lib.SubstringSearch
import ProjectEulerSolutions.Lib.TupleBuckets
import ProjectEulerSolutions.Lib.MaskedStates
import Mathlib.Tactic.Linarith

set_option Elab.async false

namespace solvers.«43»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P43Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private def numbers (items : List Int) := items.map number

namespace Domain
def initialized : List Int := [2,3,5,7,11,13,17]
def bindings (address : Nat) : Mutable.Globals := [("PRIMES", .ref address)]
def env (address : Nat) : HeapCalls.Env := fun name => (bindings address).lookup name

/-- Prerequisite evaluation compares the actual shared list to a freshly allocated literal. -/
theorem result (items : List Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (numbers items))) :
    Evaluates (globals := bindings address) program (env address) Solvers.P43Ast.requires_solve_0
      heap (flag (decide (items = initialized))) (heap ++ [.list (numbers initialized)]) := by
  let afterLiteral := heap ++ [.list (numbers initialized)]
  have original : afterLiteral[address]? = some (.list (numbers items)) :=
    ProjectEulerSolutions.HeapCollection.prefix_found heap _ address _ found
  have fresh : afterLiteral[heap.length]? = some (.list (numbers initialized)) := by simp [afterLiteral]
  have different : address ≠ heap.length := by
    have inside := (List.getElem?_eq_some_iff.mp found).1
    omega
  apply Evaluates.compare (av := .ref address) (bv := .ref heap.length)
    (middle := heap) (afterRight := afterLiteral)
    (Evaluates.lookup rfl)
    (Evaluates.list (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant
      (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant
        (.cons Evaluates.constant .nil))))))))
  simp [HeapCalls.compare, Mutable.array, original, fresh, different, numbers,
    equalLists_ints]
  exact ⟨rfl, rfl⟩

/-- A successful prerequisite pins the contents while retaining the original list identity. -/
theorem initialized_of_condition (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]")
    (condition : HeapCalls.Condition program (env address) Solvers.P43Ast.requires_solve_0 heap (bindings address)) :
    heap[address]? = some (.list (numbers initialized)) := by
  obtain ⟨items, found⟩ := Mutable.Value.ints_of_hasType typed
  have accepted := (result items heap address found).condition_iff.mp condition
  have same : items = initialized := by simpa [flag] using accepted
  simpa only [same] using found
end Domain

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    (HeapCalls.evaluate program (Domain.env address) Solvers.P43Ast.requires_solve_0 heap
      (Domain.bindings address)).Supported := by
  obtain ⟨items, found⟩ := Mutable.Value.ints_of_hasType typed
  exact Execution.supported_of_finishes (Domain.result items heap address found).finishes


namespace BuildMaps
open ProjectEulerSolutions.SubstringSearch
private abbrev natural (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev digitValues (x : Nat) := [natural (hundreds x), natural (tens x), natural (units x)]
private def digitExpressions : List Expr :=
  [.binOp (.name "x") .floorDiv (.constant (.int 100)),
   .binOp (.binOp (.name "x") .floorDiv (.constant (.int 10))) .mod (.constant (.int 10)),
   .binOp (.name "x") .mod (.constant (.int 10))]
private def duplicateTest : Expr := .boolOp .or
  [.compare (.name "a") [(.eq, .name "b")], .compare (.name "a") [(.eq, .name "c")],
    .compare (.name "b") [(.eq, .name "c")]]
def rowBody : List Stmt :=
  [.assign [.tuple [.name "a", .name "b", .name "c"]] (.tuple digitExpressions),
   .ifThen duplicateTest [.continueLoop] [], .expr ProjectEulerSolutions.TupleBuckets.insertion]

def sourceRowBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "build_maps").getD ([], [])).2).drop 1 with
  | .forIn _ _ (_ :: .forIn _ _ body _ :: _) _ :: _ => body
  | _ => []

theorem rowBody_source : rowBody = sourceRowBody := rfl

def sourceSetup : List Stmt :=
  match (executableBody ((findFunctionIn program.body "build_maps").getD ([], [])).2).drop 1 with
  | .forIn _ _ (setup :: _) _ :: _ => [setup]
  | _ => []

private def Iterates (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap) (globals : Mutable.Globals := []) : Prop :=
  ∀ next following anchors stack, Reaches (globals := globals) program anchors
    ⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩
    ⟨.block (next :: following) finalEnv, stack, finalHeap⟩

variable {globals : Mutable.Globals}

theorem setup_result (heap : Mutable.Heap) (env : HeapCalls.Env) :
    BlockContinues (globals := globals) program env sourceSetup heap (setLocal env "mp" (.ref heap.length))
      (heap ++ [.dict []]) :=
  BlockContinues.annAssign (Evaluates.dictionary .nil)

private theorem Iterates.done (env : HeapCalls.Env) (heap : Mutable.Heap) :
    Iterates (globals := globals) env [] heap env heap := by
  intro next following anchors stack
  exact .next rfl .refl

private theorem Iterates.prepend
    (first : BlockContinues (globals := globals) program env body heap middleEnv middleHeap)
    (rest : Iterates (globals := globals) middleEnv tail middleHeap finalEnv finalHeap) :
    Iterates (globals := globals) env (body ++ tail) heap finalEnv finalHeap := by
  intro next following anchors stack
  have tailRun : Reaches (globals := globals) program anchors
      ⟨.block (tail ++ (.loopResume next :: following)) middleEnv, stack, middleHeap⟩
      ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ := by
    simpa only [List.append_assoc, List.singleton_append] using rest next following anchors stack
  simpa only [List.append_assoc, List.singleton_append] using
    (first (tail ++ (.loopResume next :: following)) anchors stack).trans tailRun

private theorem Iterates.branch {flag : Bool}
    (evaluated : Evaluates (globals := globals) program env condition heap value middle)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : Iterates (globals := globals) env ((if flag then yes else no) ++ body) middle finalEnv finalHeap) :
    Iterates (globals := globals) env (.ifThen condition yes no :: body) heap finalEnv finalHeap := by
  intro next following anchors stack
  exact .next rfl ((evaluated anchors _).trans
    (.next (by simp [HeapCalls.step, HeapCalls.run, truth, List.append_assoc])
      (rest next following anchors stack)))

private theorem Iterates.skip
    (target : ∀ next following, Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following)) :
    Iterates (globals := globals) env (.continueLoop :: body) heap env heap := by
  intro next following anchors stack
  have found : Calls.loopTarget (body ++ (.loopResume next :: following)) = some (next, following) := by
    simpa only [List.append_assoc, List.singleton_append] using target next following
  exact .next (by simp [HeapCalls.step, found]) .refl

private theorem equality (left right : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (a b : String) (getA : env a = some (natural left)) (getB : env b = some (natural right)) :
    Evaluates (globals := globals) program env (.compare (.name a) [(.eq, .name b)]) heap
      (.scalar (.bool (decide (left = right)))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (Evaluates.lookup getA) (Evaluates.lookup getB)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

private theorem duplicate_result (a b c : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getA : env "a" = some (natural a)) (getB : env "b" = some (natural b))
    (getC : env "c" = some (natural c)) :
    Evaluates (globals := globals) program env duplicateTest heap
      (.scalar (.bool (decide (a = b ∨ a = c ∨ b = c)))) heap := by
  have ab := equality (globals := globals) a b heap env "a" "b" getA getB
  have ac := equality (globals := globals) a c heap env "a" "c" getA getC
  have bc := equality (globals := globals) b c heap env "b" "c" getB getC
  by_cases first : a = b
  · simp only [first, true_or, decide_true]
    exact Evaluates.boolean_stop (flag := true) (by simpa only [first, decide_true] using ab) rfl rfl
  · simp only [first, false_or]
    apply Evaluates.boolean_next (flag := false) (by simpa only [first, decide_false] using ab) rfl rfl
    by_cases second : a = c
    · simp only [second, true_or, decide_true]
      exact Evaluates.boolean_stop (flag := true) (by simpa only [second, decide_true] using ac) rfl rfl
    · simp only [second, false_or]
      apply Evaluates.boolean_next (flag := false) (by simpa only [second, decide_false] using ac) rfl rfl
      exact Evaluates.boolean_last bc rfl

def decodedEnv (env : HeapCalls.Env) (x : Nat) : HeapCalls.Env := assignNames env ["a","b","c"] (digitValues x)
def decodedHeap (heap : Mutable.Heap) (x : Nat) : Mutable.Heap := heap ++ [.tuple (digitValues x)]

def rowHeap (heap : Mutable.Heap) (dictionary : Nat) (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (x : Nat) : Mutable.Heap :=
  if distinct x then ProjectEulerSolutions.TupleBuckets.updated (decodedHeap heap x) dictionary buckets
    (tens x, units x) (hundreds x) else decodedHeap heap x

def rowBuckets (heap : Mutable.Heap) (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (x : Nat) : List ProjectEulerSolutions.TupleBuckets.Bucket :=
  if distinct x then ProjectEulerSolutions.TupleBuckets.push buckets (tens x, units x)
    (decodedHeap heap x).length ((decodedHeap heap x).length+1) (hundreds x) else buckets

private theorem decode_result (x : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getX : env "x" = some (natural x)) :
    BlockContinues (globals := globals) program env
      [.assign [.tuple [.name "a", .name "b", .name "c"]] (.tuple digitExpressions)]
      heap (decodedEnv env x) (decodedHeap heap x) := by
  have hundredsResult : Evaluates (globals := globals) program env
      (.binOp (.name "x") .floorDiv (.constant (.int 100))) heap (natural (hundreds x)) heap :=
    Evaluates.binary (Evaluates.lookup getX) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, natural, hundreds])
  have tensResult : Evaluates (globals := globals) program env
      (.binOp (.binOp (.name "x") .floorDiv (.constant (.int 10))) .mod (.constant (.int 10)))
      heap (natural (tens x)) heap := by
    apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (av := natural (x/10))
      (Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
        (Evaluates.lookup getX) Evaluates.constant (by simp [Mutable.binary, scalarBinary, natural]))
      Evaluates.constant
    simp [Mutable.binary, scalarBinary, natural, tens]
  have unitsResult : Evaluates (globals := globals) program env
      (.binOp (.name "x") .mod (.constant (.int 10))) heap (natural (units x)) heap :=
    Evaluates.binary (Evaluates.lookup getX) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, natural, units])
  apply BlockContinues.unpack (names := ["a","b","c"]) (items := digitValues x) rfl
    (Evaluates.tuple (.cons hundredsResult (.cons tensResult (.cons unitsResult .nil))) (by simp))
    (by simp [Mutable.sequence, decodedHeap]) rfl

/-- One exact source iteration, including allocation, conditional continuation, and the retained bucket lists. -/
theorem row (x : Nat) (heap : Mutable.Heap) (dictionary : Nat)
    (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (view : ProjectEulerSolutions.TupleBuckets.View heap dictionary buckets) (env : HeapCalls.Env)
    (getMap : env "mp" = some (.ref dictionary)) (getX : env "x" = some (natural x)) :
    Iterates (globals := globals) env rowBody heap (decodedEnv env x) (rowHeap heap dictionary buckets x) ∧
      ProjectEulerSolutions.TupleBuckets.View (rowHeap heap dictionary buckets x) dictionary (rowBuckets heap buckets x) := by
  have decoded := decode_result (globals := globals) x heap env getX
  have currentView := view.append [.tuple (digitValues x)]
  have tested := duplicate_result (globals := globals) (hundreds x) (tens x) (units x)
    (decodedHeap heap x) (decodedEnv env x)
    (by simp [decodedEnv, assignNames, setLocal]) (by simp [decodedEnv, assignNames, setLocal])
    (by simp [decodedEnv, assignNames, setLocal])
  have testValue : decide (hundreds x = tens x ∨ hundreds x = units x ∨ tens x = units x) = !(distinct x) := by
    by_cases ab : hundreds x = tens x <;> by_cases ac : hundreds x = units x <;>
      by_cases bc : tens x = units x <;> simp [distinct, ab, ac, bc]
  cases accepted : distinct x with
  | false =>
      simp only [rowHeap, rowBuckets, accepted, Bool.false_eq_true, if_false]
      refine ⟨?_, currentView⟩
      apply Iterates.prepend decoded
      apply Iterates.branch (flag := true) (by simpa only [testValue, accepted, Bool.not_false] using tested) rfl
      exact Iterates.skip (by intros; rfl)
  | true =>
      simp only [rowHeap, rowBuckets, accepted, if_true]
      refine ⟨?_, ProjectEulerSolutions.TupleBuckets.updated_view _ dictionary buckets currentView _ _⟩
      have appended := ProjectEulerSolutions.TupleBuckets.evaluate (globals := globals) program (decodedEnv env x)
        (decodedHeap heap x) dictionary buckets currentView (tens x, units x) (hundreds x)
        (by simpa [decodedEnv, assignNames, setLocal] using getMap)
        (by simp [decodedEnv, assignNames, setLocal]; rfl)
        (by simp [decodedEnv, assignNames, setLocal]; rfl)
        (by simp [decodedEnv, assignNames, setLocal]; rfl)
      apply Iterates.prepend decoded
      apply Iterates.branch (flag := false) (by simpa only [testValue, accepted, Bool.not_true] using tested) rfl
      change Iterates (globals := globals) (decodedEnv env x) [.expr ProjectEulerSolutions.TupleBuckets.insertion]
        (decodedHeap heap x) (decodedEnv env x)
        (ProjectEulerSolutions.TupleBuckets.updated (decodedHeap heap x) dictionary buckets (tens x, units x) (hundreds x))
      exact Iterates.prepend (BlockContinues.discard appended) (Iterates.done _ _)

/-- The row adds precisely its retained hundreds digit to the matching pair of trailing digits. -/
theorem row_contents (heap : Mutable.Heap) (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (x : Nat) (query : ProjectEulerSolutions.TupleBuckets.Key) :
    ProjectEulerSolutions.TupleBuckets.contents (rowBuckets heap buckets x) query =
      ProjectEulerSolutions.TupleBuckets.contents buckets query ++
        (if distinct x && decide (query = (tens x, units x)) then [hundreds x] else []) := by
  cases accepted : distinct x with
  | false => simp [rowBuckets, accepted]
  | true =>
      simp only [rowBuckets, accepted, if_true, ProjectEulerSolutions.TupleBuckets.contents_push, Bool.true_and]
      by_cases same : query = (tens x, units x)
      · subst query
        simp
      · simp [same]

/-- The input heap, including the PRIMES list and earlier tables, is an unchanged prefix. -/
theorem row_prefix (base suffix : Mutable.Heap) (dictionary : Nat)
    (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (view : ProjectEulerSolutions.TupleBuckets.View (base ++ suffix) dictionary buckets)
    (afterBase : base.length ≤ dictionary) (x : Nat) :
    ∃ extra, rowHeap (base ++ suffix) dictionary buckets x = base ++ extra := by
  cases accepted : distinct x with
  | false =>
      exact ⟨suffix ++ [.tuple (digitValues x)], by simp [rowHeap, accepted, decodedHeap, List.append_assoc]⟩
  | true =>
      have appended := view.append [.tuple (digitValues x)]
      have normalized : ProjectEulerSolutions.TupleBuckets.View
          (base ++ (suffix ++ [.tuple (digitValues x)])) dictionary buckets := by
        simpa only [List.append_assoc] using appended
      simpa only [rowHeap, accepted, if_true, decodedHeap, List.append_assoc] using
        ProjectEulerSolutions.TupleBuckets.updated_prefix base (suffix ++ [.tuple (digitValues x)]) dictionary
          buckets normalized afterBase (tens x, units x) (hundreds x)

private theorem Iterates.then_loop
    (iteration : Iterates (globals := globals) env body heap middleEnv middleHeap)
    (rest : BlockContinues (globals := globals) program middleEnv [next] middleHeap finalEnv finalHeap) :
    BlockContinues (globals := globals) program env (body ++ [.loopResume next]) heap finalEnv finalHeap := by
  intro following anchors stack
  exact (iteration next following anchors stack).trans (rest following anchors stack)

def collected (xs : List Nat) (query : ProjectEulerSolutions.TupleBuckets.Key) : List Nat :=
  (xs.filter fun x => distinct x && decide (query = (tens x, units x))).map hundreds

private theorem collected_cons (first : Nat) (rest : List Nat) (query : ProjectEulerSolutions.TupleBuckets.Key) :
    collected (first :: rest) query =
      (if distinct first && decide (query = (tens first, units first)) then [hundreds first] else []) ++ collected rest query := by
  unfold collected
  cases accepted : distinct first && decide (query = (tens first, units first)) <;> simp [accepted]

private theorem decoded_other (env : HeapCalls.Env) (x : Nat) (name : String)
    (unrelated : name ∉ ["a","b","c"]) : decodedEnv env x name = env name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at unrelated
  simp [decodedEnv, assignNames, setLocal, unrelated.1, unrelated.2.1, unrelated.2.2]

/-- Execute every remaining multiple, preserving the input prefix and all non-loop locals. -/
theorem range_result (p stop : Nat) (positive : 0 < p) (lower : 1000 ≤ stop) (upper : stop ≤ 999+p)
    (base : Mutable.Heap) (dictionary : Nat) (afterBase : base.length ≤ dictionary)
    (count first : Nat) (equation : first + count*p = stop)
    (suffix : Mutable.Heap) (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (view : ProjectEulerSolutions.TupleBuckets.View (base ++ suffix) dictionary buckets)
    (env : HeapCalls.Env) (getMap : env "mp" = some (.ref dictionary)) :
    ∃ extra finalEnv finalBuckets,
      BlockContinues (globals := globals) program env [.rangeStepLoop "x" first 1000 p rowBody []]
        (base ++ suffix) finalEnv (base ++ extra) ∧
      ProjectEulerSolutions.TupleBuckets.View (base ++ extra) dictionary finalBuckets ∧
      (∀ name, name ∉ ["x","a","b","c"] → finalEnv name = env name) ∧
      (∀ query, ProjectEulerSolutions.TupleBuckets.contents finalBuckets query =
        ProjectEulerSolutions.TupleBuckets.contents buckets query ++ collected (List.range' first count p) query) := by
  induction count generalizing first suffix buckets env with
  | zero =>
      have outside : 1000 ≤ first := by simpa using lower.trans_eq equation.symm
      refine ⟨suffix, env, buckets, ?_, view, ?_, ?_⟩
      · apply BlockContinues.range_done (by simp [Calls.rangeContinues, positive]; omega)
        exact BlockContinues.nil
      · intros; rfl
      · intro query
        simp [collected]
  | succ count ih =>
      have within : first < 1000 := by nlinarith [Nat.zero_le (count*p)]
      let bound := setLocal env "x" (natural first)
      obtain ⟨iteration, nextView⟩ := row (globals := globals) first (base ++ suffix) dictionary buckets view bound
        (by simpa [bound, setLocal] using getMap) (by simp [bound, setLocal])
      obtain ⟨nextSuffix, sameHeap⟩ := row_prefix base suffix dictionary buckets view afterBase first
      rw [sameHeap] at iteration nextView
      have getNextMap : decodedEnv bound first "mp" = some (.ref dictionary) := by
        simpa [decodedEnv, assignNames, setLocal, bound] using getMap
      obtain ⟨extra, finalEnv, finalBuckets, executed, finalView, untouched, contents⟩ :=
        ih (first+p) (by nlinarith [equation]) nextSuffix (rowBuckets (base ++ suffix) buckets first)
          nextView (decodedEnv bound first) getNextMap
      refine ⟨extra, finalEnv, finalBuckets, ?_, finalView, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues, positive]; omega)
        have followed := Iterates.then_loop iteration executed
        simpa only [Calls.loopContinuation, show Calls.loopControl rowBody = true from rfl,
          if_true, Int.natCast_add] using followed
      · intro name unrelated
        have notX : name ≠ "x" := by intro same; subst name; simp at unrelated
        have others : name ∉ ["a","b","c"] := fun member => unrelated (List.mem_cons_of_mem _ member)
        rw [untouched name unrelated, decoded_other bound first name others]
        simp [bound, setLocal, notX]
      · intro query
        rw [contents query, row_contents, List.range'_succ, collected_cons]
        exact List.append_assoc _ _ _


def sourceTableLoop : List Stmt :=
  match (executableBody ((findFunctionIn program.body "build_maps").getD ([], [])).2).drop 1 with
  | .forIn _ _ (_ :: loop :: _) _ :: _ => [loop]
  | _ => []

theorem sourceTableLoop_eq : sourceTableLoop =
    [.forIn (.name "x") (.call (.name "range") [.constant (.int 0), .constant (.int 1000), .name "p"]) rowBody []] := rfl

private theorem range_scaled (p first count : Nat) :
    List.range' first count p = (List.range count).map (fun i => first+p*i) := by
  induction count generalizing first with
  | zero => simp
  | succ count ih =>
      rw [List.range'_succ, ih, List.range_succ_eq_map, List.map_cons, List.map_map]
      congr 1
      apply List.map_congr_left
      intro i _
      simp [Nat.mul_succ, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

private theorem collected_multiples (p : Nat) (query : ProjectEulerSolutions.TupleBuckets.Key) :
    collected (multiples p) query = table p query.1 query.2 := by
  rcases query with ⟨b,c⟩
  simp only [collected, table, Prod.mk.injEq, eq_comm]

/-- Construct one complete prime table from an arbitrary initial heap and positive modulus. -/
theorem table_result (p : Nat) (positive : 0 < p) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getP : env "p" = some (natural p)) :
    ∃ extra finalEnv buckets,
      BlockContinues (globals := globals) program env (sourceSetup ++ sourceTableLoop) heap finalEnv (heap ++ extra) ∧
      ProjectEulerSolutions.TupleBuckets.View (heap ++ extra) heap.length buckets ∧
      finalEnv "mp" = some (.ref heap.length) ∧
      (∀ name, name ∉ ["mp","x","a","b","c"] → finalEnv name = env name) ∧
      (∀ query, ProjectEulerSolutions.TupleBuckets.contents buckets query = table p query.1 query.2) := by
  let count := 999/p+1
  let stop := count*p
  have remBound := Nat.mod_lt 999 positive
  have quotient := Nat.mod_add_div 999 p
  have lower : 1000 ≤ stop := by dsimp [stop, count]; nlinarith
  have upper : stop ≤ 999+p := by dsimp [stop, count]; nlinarith [Nat.div_mul_le_self 999 p]
  let ready := setLocal env "mp" (.ref heap.length)
  obtain ⟨extra, finalEnv, buckets, executed, finalView, untouched, contents⟩ :=
    range_result (globals := globals) p stop positive lower upper heap heap.length (by omega)
      count 0 (by simp [stop]) [.dict []] [] (ProjectEulerSolutions.TupleBuckets.empty_view heap)
      ready (by simp [ready, setLocal])
  have loop : BlockContinues (globals := globals) program ready sourceTableLoop
      (heap ++ [.dict []]) finalEnv (heap ++ extra) := by
    rw [sourceTableLoop_eq]
    apply BlockContinues.for_range
      (.cons Evaluates.constant (.cons Evaluates.constant
        (.cons (Evaluates.lookup (by simpa [ready, setLocal] using getP)) .nil)))
      (by
        change ((if (p : Int) = 0 then Except.error { typeName := "ValueError", args := [] }
          else Except.ok ((0 : Int), (1000 : Int), (p : Int))) : Except ExceptionObject (Int × Int × Int)) = _
        simp only [show (p : Int) ≠ 0 by omega, if_false]
        rfl) executed
  have sameRange : List.range' 0 count p = multiples p := by
    simpa only [multiples, count, Nat.zero_add] using range_scaled p 0 count
  refine ⟨extra, finalEnv, buckets, (setup_result heap env).append loop, finalView, ?_, ?_, ?_⟩
  · rw [untouched "mp" (by decide)]
    simp [ready, setLocal]
  · intro name unrelated
    have notMp : name ≠ "mp" := by intro same; subst name; simp at unrelated
    have others : name ∉ ["x","a","b","c"] := fun member => unrelated (List.mem_cons_of_mem _ member)
    rw [untouched name others]
    simp [ready, setLocal, notMp]
  · intro query
    simpa only [sameRange, ProjectEulerSolutions.TupleBuckets.contents, ProjectEulerSolutions.TupleBuckets.find,
      List.find?_nil, Option.map_none, Option.getD_none, List.nil_append, collected_multiples] using contents query


structure PrimeTable where
  modulus : Nat
  address : Nat
  buckets : List ProjectEulerSolutions.TupleBuckets.Bucket

def primeEntry (entry : PrimeTable) : Constant × Mutable.Value := (.int entry.modulus, .ref entry.address)
def primeEntries (entries : List PrimeTable) := entries.map primeEntry

def putTable : List PrimeTable → PrimeTable → List PrimeTable
  | [], entry => [entry]
  | first :: rest, entry =>
      if first.modulus = entry.modulus then entry :: rest else first :: putTable rest entry

private theorem primeEntries_put (entries : List PrimeTable) (entry : PrimeTable) :
    primeEntries (putTable entries entry) = ScalarDict.insert (primeEntries entries) (.int entry.modulus) (.ref entry.address) := by
  induction entries with
  | nil => rfl
  | cons first rest ih =>
      by_cases same : first.modulus = entry.modulus
      · simp [putTable, same, primeEntries, primeEntry, ScalarDict.insert, scalarCompare, evalCmpOp]
      · simpa [putTable, same, primeEntries, primeEntry, ScalarDict.insert, scalarCompare, evalCmpOp, Int.natCast_inj] using
          congrArg (List.cons (primeEntry first)) ih

private theorem putTable_member (entries : List PrimeTable) (entry current : PrimeTable)
    (member : current ∈ putTable entries entry) : current = entry ∨ current ∈ entries := by
  induction entries with
  | nil => simpa [putTable] using member
  | cons first rest ih =>
      by_cases same : first.modulus = entry.modulus
      · simp only [putTable, if_pos same, List.mem_cons] at member
        rcases member with equal | inside
        · exact .inl equal
        · exact .inr (List.mem_cons_of_mem _ inside)
      · simp only [putTable, if_neg same, List.mem_cons] at member
        rcases member with equal | inside
        · exact .inr (List.mem_cons.mpr (.inl equal))
        · rcases ih inside with equal | inside
          · exact .inl equal
          · exact .inr (List.mem_cons_of_mem _ inside)

structure MapsView (heap : Mutable.Heap) (address : Nat) (entries : List PrimeTable) : Prop where
  found : heap[address]? = some (.dict (primeEntries entries))
  tables : ∀ entry ∈ entries, address < entry.address ∧
    ProjectEulerSolutions.TupleBuckets.View heap entry.address entry.buckets ∧
    ∀ query, ProjectEulerSolutions.TupleBuckets.contents entry.buckets query = table entry.modulus query.1 query.2

theorem MapsView.append {heap : Mutable.Heap} {address : Nat} {entries : List PrimeTable}
    (view : MapsView heap address entries) (extra : Mutable.Heap) : MapsView (heap ++ extra) address entries := by
  constructor
  · exact ProjectEulerSolutions.HeapCollection.prefix_found heap extra address _ view.found
  · intro entry member
    obtain ⟨after, stored, correct⟩ := view.tables entry member
    exact ⟨after, stored.append extra, correct⟩

private theorem view_set_before {heap : Mutable.Heap} {dictionary : Nat}
    {buckets : List ProjectEulerSolutions.TupleBuckets.Bucket}
    (view : ProjectEulerSolutions.TupleBuckets.View heap dictionary buckets) (index : Nat)
    (before : index < dictionary) (object : Mutable.Object) :
    ProjectEulerSolutions.TupleBuckets.View (heap.set index object) dictionary buckets := by
  constructor
  · simpa only [List.getElem?_set_ne (by omega : index ≠ dictionary)] using view.found
  · intro bucket member
    have after := view.afterDictionary bucket member
    simpa only [List.getElem?_set_ne (by omega : index ≠ bucket.address)] using view.lists bucket member
  · exact view.afterDictionary
  · exact view.unique

theorem MapsView.store {heap : Mutable.Heap} {address : Nat} {entries : List PrimeTable}
    (view : MapsView heap address entries) (entry : PrimeTable) (after : address < entry.address)
    (stored : ProjectEulerSolutions.TupleBuckets.View heap entry.address entry.buckets)
    (correct : ∀ query, ProjectEulerSolutions.TupleBuckets.contents entry.buckets query = table entry.modulus query.1 query.2) :
    MapsView (heap.set address (.dict (primeEntries (putTable entries entry)))) address (putTable entries entry) := by
  constructor
  · exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp view.found).1
  · intro current member
    rcases putTable_member entries entry current member with rfl | old
    · exact ⟨after, view_set_before stored address after _, correct⟩
    · obtain ⟨afterCurrent, oldView, oldCorrect⟩ := view.tables current old
      exact ⟨afterCurrent, view_set_before oldView address afterCurrent _, oldCorrect⟩


def sourcePrimeBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "build_maps").getD ([], [])).2).drop 1 with
  | .forIn _ _ body _ :: _ => body
  | _ => []

private def storeTable : Stmt := .assign [.subscript (.name "maps") (.name "p")] (.name "mp")

theorem sourcePrimeBody_eq : sourcePrimeBody = sourceSetup ++ sourceTableLoop ++ [storeTable] := rfl

private theorem moduli_put (entries : List PrimeTable) (entry : PrimeTable) (p : Nat) :
    p ∈ (putTable entries entry).map PrimeTable.modulus ↔
      p = entry.modulus ∨ p ∈ entries.map PrimeTable.modulus := by
  induction entries with
  | nil => simp [putTable]
  | cons first rest ih =>
      by_cases same : first.modulus = entry.modulus
      · simp [putTable, same]
      · simp only [putTable, if_neg same, List.map_cons, List.mem_cons, ih]
        tauto

/-- A prime iteration creates a fresh table and stores its actual address in the outer dictionary. -/
theorem prime_result (p : Nat) (positive : 0 < p) (base suffix : Mutable.Heap) (address : Nat)
    (afterBase : base.length ≤ address) (entries : List PrimeTable)
    (view : MapsView (base ++ suffix) address entries) (env : HeapCalls.Env)
    (getMaps : env "maps" = some (.ref address)) (getP : env "p" = some (natural p)) :
    ∃ extra finalEnv entry, entry.modulus = p ∧
      BlockContinues (globals := globals) program env sourcePrimeBody (base ++ suffix) finalEnv (base ++ extra) ∧
      MapsView (base ++ extra) address (putTable entries entry) ∧
      (∀ name, name ∉ ["mp","x","a","b","c"] → finalEnv name = env name) := by
  obtain ⟨tableExtra, finalEnv, buckets, constructed, bucketView, getMp, untouched, contents⟩ :=
    table_result (globals := globals) p positive (base ++ suffix) env getP
  let entry : PrimeTable := ⟨p, (base ++ suffix).length, buckets⟩
  let object := Mutable.Object.dict (primeEntries (putTable entries entry))
  let extra := (suffix ++ tableExtra).set (address-base.length) object
  have storedHeap : ((base ++ suffix) ++ tableExtra).set address object = base ++ extra := by
    simp only [List.append_assoc, List.set_append_right _ _ afterBase, extra]
  have original := view.append tableExtra
  have after : address < entry.address := (List.getElem?_eq_some_iff.mp view.found).1
  have changed := original.store entry after bucketView contents
  rw [storedHeap] at changed
  have stored : BlockContinues (globals := globals) program finalEnv [storeTable]
      ((base ++ suffix) ++ tableExtra) finalEnv (base ++ extra) := by
    apply BlockContinues.setItem (by intros; simp)
      (Evaluates.lookup getMp)
      (.cons (Evaluates.lookup (by rw [untouched "maps" (by decide)]; exact getMaps))
        (.cons (Evaluates.lookup (by rw [untouched "p" (by decide)]; exact getP)) .nil))
    have written := Mutable.setItem_dictionary ((base ++ suffix) ++ tableExtra) address (primeEntries entries)
      original.found (.int p) (.ref (base ++ suffix).length)
    rw [← primeEntries_put entries entry, storedHeap] at written
    exact written
  refine ⟨extra, finalEnv, entry, rfl, ?_, changed, untouched⟩
  rw [sourcePrimeBody_eq]
  exact constructed.append stored


/-- Iterate over the shared modulus list without changing it or earlier heap objects. -/
theorem outer_result (all : List Nat) (base : Mutable.Heap) (primeAddress address : Nat)
    (foundPrimes : base[primeAddress]? = some (.list (all.map natural))) (afterBase : base.length ≤ address)
    (remaining : List Nat) (position : Nat) (split : all.drop position = remaining)
    (positive : ∀ p ∈ remaining, 0 < p) (suffix : Mutable.Heap) (entries : List PrimeTable)
    (view : MapsView (base ++ suffix) address entries) (env : HeapCalls.Env)
    (getMaps : env "maps" = some (.ref address)) :
    ∃ extra finalEnv finalEntries,
      BlockContinues (globals := globals) program env
        [.sequenceStepLoop "p" (.ref primeAddress) position sourcePrimeBody []]
        (base ++ suffix) finalEnv (base ++ extra) ∧
      MapsView (base ++ extra) address finalEntries ∧
      (∀ name, name ∉ ["p","mp","x","a","b","c"] → finalEnv name = env name) ∧
      (∀ p, p ∈ finalEntries.map PrimeTable.modulus ↔ p ∈ entries.map PrimeTable.modulus ∨ p ∈ remaining) := by
  have sequence (extra : Mutable.Heap) : Mutable.sequence (.ref primeAddress) (base ++ extra) =
      some (.ok (all.map natural), base ++ extra) := by
    have found := ProjectEulerSolutions.HeapCollection.prefix_found base extra primeAddress _ foundPrimes
    simp [Mutable.sequence, found]
  induction remaining generalizing position suffix entries env with
  | nil =>
      refine ⟨suffix, env, entries, ?_, view, ?_, ?_⟩
      · apply BlockContinues.sequence_done (source := .ref primeAddress) (sequence suffix)
        have missing : all[position]? = none := by
          simpa only [split, List.head?_nil] using (List.head?_drop (l := all) (i := position)).symm
        · simp [missing]
        · exact BlockContinues.nil
      · intros; rfl
      · intro p; simp
  | cons p remaining ih =>
      let bound := setLocal env "p" (natural p)
      obtain ⟨nextSuffix, middleEnv, entry, primeEqual, step, nextView, same⟩ :=
        prime_result (globals := globals) p (positive p List.mem_cons_self) base suffix address afterBase entries view bound
          (by simpa [bound, setLocal] using getMaps) (by simp [bound, setLocal])
      have nextMaps : middleEnv "maps" = some (.ref address) := by
        rw [same "maps" (by decide)]
        simpa [bound, setLocal] using getMaps
      obtain ⟨extra, finalEnv, finalEntries, executed, finalView, untouched, covered⟩ :=
        ih (position+1) (by simp [List.drop_add_one_eq_tail_drop, split])
          (fun q member => positive q (List.mem_cons_of_mem _ member)) nextSuffix (putTable entries entry)
          nextView middleEnv nextMaps
      refine ⟨extra, finalEnv, finalEntries, ?_, finalView, ?_, ?_⟩
      · have selected : all[position]? = some p := by
          simpa only [split, List.head?_cons] using (List.head?_drop (l := all) (i := position)).symm
        apply BlockContinues.sequence_step (source := .ref primeAddress) (value := natural p) (sequence suffix) (by simp [selected])
        simpa only [Calls.loopContinuation, show Calls.loopControl sourcePrimeBody = false from rfl,
          Bool.false_eq_true, if_false] using step.append executed
      · intro name unrelated
        have notP : name ≠ "p" := by intro equal; subst name; simp at unrelated
        have others : name ∉ ["mp","x","a","b","c"] := fun member => unrelated (List.mem_cons_of_mem _ member)
        rw [untouched name unrelated, same name others]
        simp [bound, setLocal, notP]
      · intro q
        rw [covered q, moduli_put, primeEqual]
        simp only [List.mem_cons, or_assoc, or_left_comm, or_comm]

def sourceMapsSetup : List Stmt :=
  (executableBody ((findFunctionIn program.body "build_maps").getD ([], [])).2).take 1

private theorem maps_setup (heap : Mutable.Heap) (env : HeapCalls.Env) :
    BlockContinues (globals := globals) program env sourceMapsSetup heap
      (setLocal env "maps" (.ref heap.length)) (heap ++ [.dict []]) :=
  BlockContinues.annAssign (Evaluates.dictionary .nil)

/-- The complete table builder returns every requested modulus table and preserves the input heap. -/
theorem result (primes : List Nat) (positive : ∀ p ∈ primes, 0 < p) (heap : Mutable.Heap) (primeAddress : Nat)
    (foundPrimes : heap[primeAddress]? = some (.list (primes.map natural))) :
    ∃ extra entries, ReturnsValue (globals := Domain.bindings primeAddress) program "build_maps" [] heap
        (.ref heap.length) (heap ++ extra) ∧ MapsView (heap ++ extra) heap.length entries ∧
      (∀ p, p ∈ entries.map PrimeTable.modulus ↔ p ∈ primes) := by
  let initial := functionEnv ((findFunctionIn program.body "build_maps").getD ([], [])).1
    ((findFunctionIn program.body "build_maps").getD ([], [])).2 [] (Domain.bindings primeAddress)
  let ready := setLocal initial "maps" (.ref heap.length)
  have initialized : MapsView (heap ++ [.dict []]) heap.length [] := by
    constructor
    · simp [primeEntries]
    · simp
  obtain ⟨extra, finalEnv, entries, executed, finalView, untouched, covered⟩ :=
    outer_result (globals := Domain.bindings primeAddress) primes heap primeAddress heap.length foundPrimes (by omega)
      primes 0 rfl positive [.dict []] [] initialized ready (by simp [ready, setLocal])
  have loop : BlockContinues (globals := Domain.bindings primeAddress) program ready
      [.forIn (.name "p") (.name "PRIMES") sourcePrimeBody []] (heap ++ [.dict []]) finalEnv (heap ++ extra) := by
    apply BlockContinues.for_sequence (value := .ref primeAddress) (items := primes.map natural) (by intros; simp)
      (Evaluates.lookup (by
        have noArgs : ((findFunctionIn program.body "build_maps").getD ([], [])).1 = [] := rfl
        have notLocal : "PRIMES" ∉ blockLocals ((findFunctionIn program.body "build_maps").getD ([], [])).2 := by
          simp [program, Solvers.P43Ast.module, findFunctionIn, blockLocals, assignedLocals, targetLocals]
        simp [ready, setLocal, initial, functionEnv, Domain.bindings, noArgs, notLocal]))
      (by have stored := ProjectEulerSolutions.HeapCollection.prefix_found heap [.dict []] primeAddress _ foundPrimes
          simp [Mutable.sequence, stored]) executed
  refine ⟨extra, entries, ?_, finalView, ?_⟩
  · apply BlockReturns.function rfl rfl
    apply ((maps_setup heap initial).append loop).returns
    apply BlockReturns.return_value
    apply Evaluates.lookup
    rw [untouched "maps" (by decide)]
    simp [ready, setLocal]
  · intro p
    simpa only [List.map_nil, List.not_mem_nil, false_or] using covered p


private theorem lookup_prime_entry (entries : List PrimeTable) (p : Nat)
    (present : p ∈ entries.map PrimeTable.modulus) :
    ∃ entry ∈ entries, entry.modulus = p ∧
      ScalarDict.lookup (primeEntries entries) (.int p) = some (.ref entry.address) := by
  induction entries with
  | nil => simp at present
  | cons first rest ih =>
      by_cases same : first.modulus = p
      · refine ⟨first, List.mem_cons_self, same, ?_⟩
        simp [primeEntries, primeEntry, ScalarDict.lookup, scalarCompare, evalCmpOp, same]
      · have tail : p ∈ rest.map PrimeTable.modulus := by
          simpa only [List.map_cons, List.mem_cons, show p ≠ first.modulus from Ne.symm same, false_or] using present
        obtain ⟨entry, inside, equal, lookup⟩ := ih tail
        refine ⟨entry, List.mem_cons_of_mem _ inside, equal, ?_⟩
        simpa [primeEntries, primeEntry, ScalarDict.lookup, scalarCompare, evalCmpOp, Int.natCast_inj, same] using lookup

/-- Subsequent source lookups retrieve the constructed table through its original dictionary reference. -/
theorem table_lookup (heap : Mutable.Heap) (address : Nat) (entries : List PrimeTable)
    (view : MapsView heap address entries) (p : Nat) (present : p ∈ entries.map PrimeTable.modulus) :
    ∃ dictionary buckets, Mutable.getItem (.ref address) (natural p) heap = some (.ok (.ref dictionary), heap) ∧
      ProjectEulerSolutions.TupleBuckets.View heap dictionary buckets ∧
      ∀ query, ProjectEulerSolutions.TupleBuckets.contents buckets query = table p query.1 query.2 := by
  obtain ⟨entry, member, equal, lookup⟩ := lookup_prime_entry entries p present
  obtain ⟨_, stored, correct⟩ := view.tables entry member
  refine ⟨entry.address, entry.buckets, ?_, stored, ?_⟩
  · exact Mutable.getItem_dictionary heap address (primeEntries entries) view.found (.int p) lookup
  · simpa only [equal] using correct

end BuildMaps

namespace SeedStates
open ProjectEulerSolutions.SubstringSearch
private abbrev natural (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev seed (x : Nat) : ProjectEulerSolutions.MaskedStates.State :=
  (digits x, (2^hundreds x) ||| (2^tens x) ||| (2^units x))
private def shifted (name : String) : Expr := .binOp (.constant (.int 1)) .leftShift (.name name)
private def maskExpression : Expr := .binOp (.binOp (shifted "a") .bitOr (shifted "b")) .bitOr (shifted "c")
private def usedAssignment : Stmt := .assign [.name "used"] maskExpression
private def digitsExpression : Expr := .tuple [.name "a", .name "b", .name "c"]
private def appendState : Stmt := .expr (.call (.attribute (.name "states") "append")
  [.tuple [digitsExpression, .name "used"]])
def body : List Stmt := BuildMaps.rowBody.take 2 ++ [usedAssignment, appendState]

def sourceBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 2 with
  | .forIn _ _ body _ :: _ => body
  | _ => []

theorem body_source : body = sourceBody := rfl

private theorem shift_result {globals : Mutable.Globals} (name : String) (n : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getName : env name = some (natural n)) :
    Evaluates (globals := globals) program env (shifted name) heap (natural (2^n)) heap := by
  apply Evaluates.binary Evaluates.constant (Evaluates.lookup getName)
  simp [Mutable.binary, scalarShiftLeft, scalarNumber, integerShiftLeft, natural, Int.not_lt.mpr (Int.natCast_nonneg n)]

private theorem mask_result {globals : Mutable.Globals} (a b c : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getA : env "a" = some (natural a)) (getB : env "b" = some (natural b)) (getC : env "c" = some (natural c)) :
    Evaluates (globals := globals) program env maskExpression heap (natural ((2^a) ||| (2^b) ||| (2^c))) heap := by
  have first : Evaluates (globals := globals) program env (.binOp (shifted "a") .bitOr (shifted "b"))
      heap (natural ((2^a) ||| (2^b))) heap :=
    Evaluates.binary (shift_result "a" a heap env getA) (shift_result "b" b heap env getB)
      (by simp [Mutable.binary, scalarBitOr, scalarNumber, natural]; rfl)
  exact Evaluates.binary first (shift_result "c" c heap env getC)
    (by simp [Mutable.binary, scalarBitOr, scalarNumber, natural]; rfl)

/-- One seed iteration executes the actual duplicate test, mask calculation and nested tuple append. -/
theorem row {globals : Mutable.Globals} (x : Nat) (base garbage : Mutable.Heap)
    (cells : List ProjectEulerSolutions.MaskedStates.Cell)
    (view : ProjectEulerSolutions.MaskedStates.View
      (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage) base.length cells)
    (env : HeapCalls.Env) (getStates : env "states" = some (.ref base.length)) (getX : env "x" = some (natural x)) :
    ∃ finalEnv finalGarbage finalCells,
      BuildMaps.Iterates (globals := globals) env body
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)
        finalEnv (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) finalGarbage) ∧
      ProjectEulerSolutions.MaskedStates.View
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) finalGarbage) base.length finalCells ∧
      finalCells.map ProjectEulerSolutions.MaskedStates.Cell.state =
        cells.map ProjectEulerSolutions.MaskedStates.Cell.state ++ (if distinct x then [seed x] else []) ∧
      (∀ name, name ∉ ["a", "b", "c", "used"] → finalEnv name = env name) := by
  let initial := ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage
  let decoded := BuildMaps.decodedEnv env x
  let decodedGarbage := garbage ++ [.tuple (BuildMaps.digitValues x)]
  let afterDecode := ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) decodedGarbage
  have decodedHeap : BuildMaps.decodedHeap initial x = afterDecode := by
    simp [BuildMaps.decodedHeap, initial, afterDecode, decodedGarbage, ProjectEulerSolutions.HeapCollection.append]
  have decode := BuildMaps.decode_result (globals := globals) x initial env getX
  rw [decodedHeap] at decode
  have decodedView : ProjectEulerSolutions.MaskedStates.View afterDecode base.length cells := by
    simpa only [← decodedHeap, BuildMaps.decodedHeap] using view.append [.tuple (BuildMaps.digitValues x)]
  have getA : decoded "a" = some (natural (hundreds x)) := by simp [decoded, BuildMaps.decodedEnv, assignNames, setLocal]
  have getB : decoded "b" = some (natural (tens x)) := by simp [decoded, BuildMaps.decodedEnv, assignNames, setLocal]
  have getC : decoded "c" = some (natural (units x)) := by simp [decoded, BuildMaps.decodedEnv, assignNames, setLocal]
  have tested := BuildMaps.duplicate_result (globals := globals) (hundreds x) (tens x) (units x) afterDecode decoded getA getB getC
  have testValue : decide (hundreds x = tens x ∨ hundreds x = units x ∨ tens x = units x) = !(distinct x) := by
    by_cases ab : hundreds x = tens x <;> by_cases ac : hundreds x = units x <;>
      by_cases bc : tens x = units x <;> simp [distinct, ab, ac, bc]
  cases accepted : distinct x with
  | false =>
      refine ⟨decoded, decodedGarbage, cells, ?_, decodedView, by simp, ?_⟩
      · apply BuildMaps.Iterates.prepend decode
        apply BuildMaps.Iterates.branch (body := [usedAssignment, appendState]) (flag := true) (by simpa only [testValue, accepted, Bool.not_false] using tested) rfl
        exact BuildMaps.Iterates.skip (by intros; rfl)
      · intro name unrelated
        have na : name ≠ "a" := by intro equal; subst name; simp at unrelated
        have nb : name ≠ "b" := by intro equal; subst name; simp at unrelated
        have nc : name ≠ "c" := by intro equal; subst name; simp at unrelated
        simp [decoded, BuildMaps.decodedEnv, assignNames, setLocal, na, nb, nc]
  | true =>
      let ready := setLocal decoded "used" (natural (seed x).2)
      have used : BlockContinues (globals := globals) program decoded [usedAssignment] afterDecode ready afterDecode :=
        BlockContinues.assign (mask_result _ _ _ afterDecode decoded getA getB getC)
      have digitTuple : Evaluates (globals := globals) program ready digitsExpression afterDecode (.ref afterDecode.length)
          (afterDecode ++ [.tuple (ProjectEulerSolutions.MaskedStates.numbers (seed x).1)]) :=
        Evaluates.tuple (.cons (Evaluates.lookup (by simpa [ready, setLocal] using getA))
          (.cons (Evaluates.lookup (by simpa [ready, setLocal] using getB))
            (.cons (Evaluates.lookup (by simpa [ready, setLocal] using getC)) .nil))) (by simp [ProjectEulerSolutions.MaskedStates.numbers, digits])
      have currentStates : ready "states" = some (.ref base.length) := by
        simpa [ready, decoded, BuildMaps.decodedEnv, assignNames, setLocal] using getStates
      have bound : base.length < afterDecode.length := by
        simp only [afterDecode, ProjectEulerSolutions.HeapCollection.working, List.length_append, List.length_cons]
        omega
      obtain ⟨appended, finalView⟩ := ProjectEulerSolutions.MaskedStates.append (globals := globals) program ready
        base decodedGarbage [.tuple (ProjectEulerSolutions.MaskedStates.numbers (seed x).1)] cells decodedView
        "states" currentStates digitsExpression (.name "used") (seed x) afterDecode.length digitTuple (by simp [afterDecode]) bound
        (Evaluates.lookup (by simp [ready, setLocal]; rfl))
      refine ⟨ready, _, _, ?_, finalView, ?_, ?_⟩
      · apply BuildMaps.Iterates.prepend decode
        apply BuildMaps.Iterates.branch (body := [usedAssignment, appendState]) (flag := false) (by simpa only [testValue, accepted, Bool.not_true] using tested) rfl
        change BuildMaps.Iterates (globals := globals) decoded [usedAssignment, appendState] afterDecode ready _
        exact BuildMaps.Iterates.prepend (used.append appended) (BuildMaps.Iterates.done _ _)
      · simp
      · intro name unrelated
        have na : name ≠ "a" := by intro equal; subst name; simp at unrelated
        have nb : name ≠ "b" := by intro equal; subst name; simp at unrelated
        have nc : name ≠ "c" := by intro equal; subst name; simp at unrelated
        have nu : name ≠ "used" := by intro equal; subst name; simp at unrelated
        simp [ready, decoded, BuildMaps.decodedEnv, assignNames, setLocal, na, nb, nc, nu]

private def collected (xs : List Nat) : List ProjectEulerSolutions.MaskedStates.State :=
  (xs.filter distinct).map seed

private theorem collected_cons (first : Nat) (rest : List Nat) :
    collected (first :: rest) = (if distinct first then [seed first] else []) ++ collected rest := by
  cases accepted : distinct first <;> simp [collected, accepted]

/-- The source seed loop visits exactly the increasing multiples of seventeen. -/
theorem range_result {globals : Mutable.Globals} (base : Mutable.Heap) (count first : Nat)
    (equation : first + count*17 = 1003) (garbage : Mutable.Heap)
    (cells : List ProjectEulerSolutions.MaskedStates.Cell)
    (view : ProjectEulerSolutions.MaskedStates.View
      (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage) base.length cells)
    (env : HeapCalls.Env) (getStates : env "states" = some (.ref base.length)) :
    ∃ finalGarbage finalEnv finalCells,
      BlockContinues (globals := globals) program env [.rangeStepLoop "x" first 1000 17 body []]
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)
        finalEnv (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) finalGarbage) ∧
      ProjectEulerSolutions.MaskedStates.View
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) finalGarbage) base.length finalCells ∧
      (∀ name, name ∉ ["x", "a", "b", "c", "used"] → finalEnv name = env name) ∧
      finalCells.map ProjectEulerSolutions.MaskedStates.Cell.state =
        cells.map ProjectEulerSolutions.MaskedStates.Cell.state ++ collected (List.range' first count 17) := by
  induction count generalizing first garbage cells env with
  | zero =>
      have outside : 1000 ≤ first := by omega
      refine ⟨garbage, env, cells, ?_, view, ?_, ?_⟩
      · exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
      · intros; rfl
      · simp [collected]
  | succ count ih =>
      have within : first < 1000 := by omega
      let bound := setLocal env "x" (natural first)
      obtain ⟨middleEnv, nextGarbage, nextCells, iteration, nextView, nextValues, same⟩ :=
        row (globals := globals) first base garbage cells view bound
          (by simpa [bound, setLocal] using getStates) (by simp [bound, setLocal])
      have nextStates : middleEnv "states" = some (.ref base.length) := by
        rw [same "states" (by decide)]
        simpa [bound, setLocal] using getStates
      obtain ⟨finalGarbage, finalEnv, finalCells, executed, finalView, untouched, values⟩ :=
        ih (first+17) (by omega) nextGarbage nextCells nextView middleEnv nextStates
      refine ⟨finalGarbage, finalEnv, finalCells, ?_, finalView, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        have followed := BuildMaps.Iterates.then_loop iteration executed
        simpa only [Calls.loopContinuation, show Calls.loopControl body = true from rfl,
          if_true, Int.natCast_add] using followed
      · intro name unrelated
        have notX : name ≠ "x" := by intro equal; subst name; simp at unrelated
        have others : name ∉ ["a", "b", "c", "used"] := fun member => unrelated (List.mem_cons_of_mem _ member)
        rw [untouched name unrelated, same name others]
        simp [bound, setLocal, notX]
      · rw [values, nextValues, List.range'_succ, collected_cons]
        exact List.append_assoc _ _ _

def sourceSetup : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 1).take 1

def sourceLoop : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 2).take 1

/-- Initialization and all fifty-nine source iterations produce precisely the mathematical seed states. -/
theorem result {globals : Mutable.Globals} (heap : Mutable.Heap) (env : HeapCalls.Env) :
    ∃ garbage finalEnv cells,
      BlockContinues (globals := globals) program env (sourceSetup ++ sourceLoop) heap finalEnv
        (ProjectEulerSolutions.HeapCollection.working heap (ProjectEulerSolutions.MaskedStates.references cells) garbage) ∧
      ProjectEulerSolutions.MaskedStates.View
        (ProjectEulerSolutions.HeapCollection.working heap (ProjectEulerSolutions.MaskedStates.references cells) garbage) heap.length cells ∧
      finalEnv "states" = some (.ref heap.length) ∧
      (∀ name, name ∉ ["states", "x", "a", "b", "c", "used"] → finalEnv name = env name) ∧
      cells.map ProjectEulerSolutions.MaskedStates.Cell.state = seeds := by
  let ready := setLocal env "states" (.ref heap.length)
  have initialized : ProjectEulerSolutions.MaskedStates.View
      (ProjectEulerSolutions.HeapCollection.working heap (ProjectEulerSolutions.MaskedStates.references []) []) heap.length [] := by
    constructor <;> simp
  obtain ⟨garbage, finalEnv, cells, executed, finalView, untouched, values⟩ :=
    range_result (globals := globals) heap 59 0 rfl [] [] initialized ready (by simp [ready, setLocal])
  have loop : BlockContinues (globals := globals) program ready sourceLoop
      (heap ++ [.list []]) finalEnv
      (ProjectEulerSolutions.HeapCollection.working heap (ProjectEulerSolutions.MaskedStates.references cells) garbage) := by
    exact BlockContinues.for_range (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))) rfl executed
  refine ⟨garbage, finalEnv, cells, (BlockContinues.annAssign (Evaluates.list .nil)).append loop, finalView, ?_, ?_, ?_⟩
  · rw [untouched "states" (by decide)]
    simp [ready, setLocal]
  · intro name unrelated
    have notStates : name ≠ "states" := by intro equal; subst name; simp at unrelated
    have others : name ∉ ["x", "a", "b", "c", "used"] := fun member => unrelated (List.mem_cons_of_mem _ member)
    rw [untouched name others]
    simp [ready, setLocal, notStates]
  · have enumeration : List.range' 0 59 17 = multiples 17 := by
      simpa only [multiples, zero_add] using BuildMaps.range_scaled 17 0 59
    simpa only [List.map_nil, List.nil_append, collected, enumeration] using values

end SeedStates

namespace ExtendStates
open ProjectEulerSolutions.SubstringSearch
private abbrev natural (n : Nat) : Mutable.Value := .scalar (.int n)
private def maskTest : Expr := .binOp (.name "used") .bitAnd (SeedStates.shifted "a")
private def digitsExpression : Expr := .binOp (.tuple [.name "a"]) .add (.name "digits")
private def maskExpression : Expr := .binOp (.name "used") .bitOr (SeedStates.shifted "a")
private def appendState : Stmt := .expr (.call (.attribute (.name "new_states") "append")
  [.tuple [digitsExpression, maskExpression]])
def candidateBody : List Stmt := [.ifThen maskTest [.continueLoop] [], appendState]

def sourceCandidateBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3 with
  | .forIn _ _ (_ :: _ :: .forIn _ _ (_ :: .forIn _ _ body _ :: _) _ :: _) _ :: _ => body
  | _ => []

theorem candidateBody_source : candidateBody = sourceCandidateBody := rfl

/-- A candidate either continues without allocation or appends the exact prepended suffix and new mask. -/
theorem candidate {globals : Mutable.Globals} (digit : Nat) (state : ProjectEulerSolutions.MaskedStates.State)
    (nonempty : state.1 ≠ []) (digitsAddress : Nat) (base garbage : Mutable.Heap)
    (cells : List ProjectEulerSolutions.MaskedStates.Cell)
    (view : ProjectEulerSolutions.MaskedStates.View
      (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage) base.length cells)
    (digitsFound : (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)[digitsAddress]? =
      some (.tuple (ProjectEulerSolutions.MaskedStates.numbers state.1)))
    (env : HeapCalls.Env) (getStates : env "new_states" = some (.ref base.length))
    (getDigits : env "digits" = some (.ref digitsAddress)) (getUsed : env "used" = some (natural state.2))
    (getA : env "a" = some (natural digit)) :
    ∃ extra finalCells,
      BuildMaps.Iterates (globals := globals) env candidateBody
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)
        env (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (garbage ++ extra)) ∧
      ProjectEulerSolutions.MaskedStates.View
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (garbage ++ extra)) base.length finalCells ∧
      finalCells.map ProjectEulerSolutions.MaskedStates.Cell.state =
        cells.map ProjectEulerSolutions.MaskedStates.Cell.state ++
          (if state.2 &&& 2^digit = 0 then [(digit :: state.1, state.2 ||| 2^digit)] else []) := by
  let initial := ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage
  have tested : Evaluates (globals := globals) program env maskTest initial (natural (state.2 &&& 2^digit)) initial :=
    Evaluates.binary (Evaluates.lookup getUsed) (SeedStates.shift_result "a" digit initial env getA)
      (by simp [Mutable.binary, scalarBitAnd, scalarNumber, natural]; rfl)
  by_cases overlap : state.2 &&& 2^digit ≠ 0
  · refine ⟨[], cells, ?_, by simpa using view, by simp [overlap]⟩
    simp only [List.append_nil]
    apply BuildMaps.Iterates.branch (body := [appendState]) (flag := true) tested
      (by simp [Mutable.truth, scalarTruth, overlap])
    exact BuildMaps.Iterates.skip (by intros; rfl)
  · have unused : state.2 &&& 2^digit = 0 := by omega
    let singleton := Mutable.Object.tuple [natural digit]
    let digitsObject := Mutable.Object.tuple (ProjectEulerSolutions.MaskedStates.numbers (digit :: state.1))
    let afterSingleton := initial ++ [singleton]
    let digitsAddress' := initial.length+1
    have single : afterSingleton[initial.length]? = some singleton := by simp [afterSingleton]
    have original : afterSingleton[digitsAddress]? = some (.tuple (ProjectEulerSolutions.MaskedStates.numbers state.1)) :=
      ProjectEulerSolutions.HeapCollection.prefix_found initial [singleton] _ _ digitsFound
    have concat : Evaluates (globals := globals) program env digitsExpression initial (.ref digitsAddress')
        (initial ++ [singleton, digitsObject]) := by
      apply Evaluates.binary (bv := .ref digitsAddress) (afterRight := afterSingleton)
        (Evaluates.tuple (.cons (Evaluates.lookup getA) .nil) (by simp)) (Evaluates.lookup getDigits)
      simp only [Mutable.binary, show (BinOp.add == BinOp.add) = true from rfl, if_true]
      simp [single, original, singleton, Mutable.allocateTuple,
        ProjectEulerSolutions.MaskedStates.numbers, List.map_eq_nil_iff, nonempty, Except.map]
      simp [afterSingleton, digitsAddress', digitsObject, singleton, ProjectEulerSolutions.MaskedStates.numbers,
        ProjectEulerSolutions.MaskedStates.number, List.append_assoc]
    have combined : Evaluates (globals := globals) program env maskExpression (initial ++ [singleton, digitsObject])
        (ProjectEulerSolutions.MaskedStates.number (state.2 ||| 2^digit)) (initial ++ [singleton, digitsObject]) :=
      Evaluates.binary (Evaluates.lookup getUsed)
        (SeedStates.shift_result "a" digit (initial ++ [singleton, digitsObject]) env getA)
        (by simp [Mutable.binary, scalarBitOr, scalarNumber, ProjectEulerSolutions.MaskedStates.number]; rfl)
    have afterCollector : base.length < digitsAddress' := by
      simp only [digitsAddress', initial, ProjectEulerSolutions.HeapCollection.working, List.length_append, List.length_cons]
      omega
    have freshDigits : (initial ++ [singleton, digitsObject])[digitsAddress']? = some digitsObject := by
      simpa only [digitsAddress', List.length_append, List.length_singleton, List.append_assoc, List.singleton_append] using
        (show ((initial ++ [singleton]) ++ [digitsObject])[(initial ++ [singleton]).length]? = some digitsObject by simp)
    obtain ⟨appended, finalView⟩ := ProjectEulerSolutions.MaskedStates.append (globals := globals) program env
      base garbage [singleton, digitsObject] cells view "new_states" getStates digitsExpression maskExpression
      (digit :: state.1, state.2 ||| 2^digit) digitsAddress' concat
      freshDigits afterCollector combined
    refine ⟨[singleton, digitsObject] ++ [.tuple [.ref digitsAddress', ProjectEulerSolutions.MaskedStates.number (state.2 ||| 2^digit)]],
      cells ++ [⟨(digit :: state.1, state.2 ||| 2^digit), digitsAddress', (initial ++ [singleton, digitsObject]).length⟩],
      ?_, ?_, ?_⟩
    · apply BuildMaps.Iterates.branch (body := [appendState]) (flag := false) tested
        (by simp [Mutable.truth, scalarTruth, unused])
      change BuildMaps.Iterates (globals := globals) env [appendState] initial env _
      exact BuildMaps.Iterates.prepend (by simpa only [List.append_assoc] using appended) (BuildMaps.Iterates.done _ _)
    · simpa only [List.append_assoc] using finalView
    · simp [unused]

private def collected (state : ProjectEulerSolutions.MaskedStates.State) (candidates : List Nat) :
    List ProjectEulerSolutions.MaskedStates.State :=
  (candidates.filter fun digit => decide (state.2 &&& 2^digit = 0)).map fun digit =>
    (digit :: state.1, state.2 ||| 2^digit)

private theorem collected_cons (state : ProjectEulerSolutions.MaskedStates.State) (digit : Nat) (rest : List Nat) :
    collected state (digit :: rest) =
      (if state.2 &&& 2^digit = 0 then [(digit :: state.1, state.2 ||| 2^digit)] else []) ++ collected state rest := by
  by_cases unused : state.2 &&& 2^digit = 0 <;> simp [collected, unused]

/-- Iterate a shared candidate list without changing it or the source suffix tuple. -/
theorem candidates_result {globals : Mutable.Globals} (all : List Nat)
    (base : Mutable.Heap) (sourceAddress : Nat) (different : base.length ≠ sourceAddress)
    (state : ProjectEulerSolutions.MaskedStates.State) (nonempty : state.1 ≠ []) (digitsAddress : Nat)
    (digitsDifferent : base.length ≠ digitsAddress)
    (remaining : List Nat) (position : Nat) (split : all.drop position = remaining)
    (garbage : Mutable.Heap) (cells : List ProjectEulerSolutions.MaskedStates.Cell)
    (view : ProjectEulerSolutions.MaskedStates.View
      (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage) base.length cells)
    (sourceFound : (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)[sourceAddress]? =
      some (.list (ProjectEulerSolutions.MaskedStates.numbers all)))
    (digitsFound : (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)[digitsAddress]? =
      some (.tuple (ProjectEulerSolutions.MaskedStates.numbers state.1)))
    (env : HeapCalls.Env) (getStates : env "new_states" = some (.ref base.length))
    (getDigits : env "digits" = some (.ref digitsAddress)) (getUsed : env "used" = some (natural state.2)) :
    ∃ extra finalEnv finalCells,
      BlockContinues (globals := globals) program env
        [.sequenceStepLoop "a" (.ref sourceAddress) position candidateBody []]
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)
        finalEnv (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (garbage ++ extra)) ∧
      ProjectEulerSolutions.MaskedStates.View
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (garbage ++ extra)) base.length finalCells ∧
      (∀ name, name ≠ "a" → finalEnv name = env name) ∧
      finalCells.map ProjectEulerSolutions.MaskedStates.Cell.state =
        cells.map ProjectEulerSolutions.MaskedStates.Cell.state ++ collected state remaining := by
  induction remaining generalizing position garbage cells env with
  | nil =>
      refine ⟨[], env, cells, ?_, by simpa using view, ?_, by simp [collected]⟩
      · simp only [List.append_nil]
        apply BlockContinues.sequence_done (source := .ref sourceAddress) (items := ProjectEulerSolutions.MaskedStates.numbers all)
          (by simp [Mutable.sequence, sequenceValue, sourceFound])
          (by have empty := congrArg List.head? split
              simpa [ProjectEulerSolutions.MaskedStates.numbers, List.head?_drop] using empty) .nil
      · intros; rfl
  | cons digit rest ih =>
      let bound := setLocal env "a" (natural digit)
      obtain ⟨extra, nextCells, iteration, nextView, nextValues⟩ :=
        candidate (globals := globals) digit state nonempty digitsAddress base garbage cells view digitsFound bound
          (by simpa [bound, setLocal] using getStates) (by simpa [bound, setLocal] using getDigits)
          (by simpa [bound, setLocal] using getUsed) (by simp [bound, setLocal])
      have sourceKept := ProjectEulerSolutions.HeapCollection.other_found base
        (ProjectEulerSolutions.MaskedStates.references cells) (ProjectEulerSolutions.MaskedStates.references nextCells)
        garbage extra sourceAddress _ different sourceFound
      have digitsKept := ProjectEulerSolutions.HeapCollection.other_found base
        (ProjectEulerSolutions.MaskedStates.references cells) (ProjectEulerSolutions.MaskedStates.references nextCells)
        garbage extra digitsAddress _ digitsDifferent digitsFound
      obtain ⟨tailExtra, finalEnv, finalCells, executed, finalView, untouched, values⟩ :=
        ih (position+1) (by simp [List.drop_add_one_eq_tail_drop, split]) (garbage ++ extra) nextCells nextView
          sourceKept digitsKept bound (by simpa [bound, setLocal] using getStates)
          (by simpa [bound, setLocal] using getDigits) (by simpa [bound, setLocal] using getUsed)
      refine ⟨extra ++ tailExtra, finalEnv, finalCells, ?_, by simpa only [List.append_assoc] using finalView, ?_, ?_⟩
      · have selected : all[position]? = some digit := by
          simpa only [split, List.head?_cons] using (List.head?_drop (l := all) (i := position)).symm
        apply BlockContinues.sequence_step (source := .ref sourceAddress) (items := ProjectEulerSolutions.MaskedStates.numbers all) (value := natural digit)
          (by simp [Mutable.sequence, sequenceValue, sourceFound])
          (by simp [ProjectEulerSolutions.MaskedStates.numbers, selected]; rfl)
        have followed := BuildMaps.Iterates.then_loop iteration executed
        simpa only [Calls.loopContinuation, show Calls.loopControl candidateBody = true from rfl,
          if_true, List.append_assoc] using followed
      · intro name unrelated
        rw [untouched name unrelated]
        simp [bound, setLocal, unrelated]
      · rw [values, nextValues, collected_cons]
        exact List.append_assoc _ _ _

private def pairExpression : Expr := .tuple
  [.subscript (.name "digits") (.constant (.int 0)), .subscript (.name "digits") (.constant (.int 1))]
def stateBody : List Stmt := [.assign [.tuple [.name "b", .name "c"]] pairExpression,
  .forIn (.name "a") ProjectEulerSolutions.TupleBuckets.lookupExpression candidateBody []]

def sourceStateBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3 with
  | .forIn _ _ (_ :: _ :: .forIn _ _ body _ :: _) _ :: _ => body
  | _ => []

theorem stateBody_source : stateBody = sourceStateBody := rfl

/-- A source state reads its first two digits and traverses exactly the matching shared bucket. -/
theorem row {globals : Mutable.Globals} (left right used : Nat) (tail : List Nat)
    (base : Mutable.Heap) (dictionary : Nat) (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (tableView : ProjectEulerSolutions.TupleBuckets.View base dictionary buckets)
    (digitsAddress : Nat) (digitsFound : base[digitsAddress]? =
      some (.tuple (ProjectEulerSolutions.MaskedStates.numbers (left :: right :: tail))))
    (garbage : Mutable.Heap) (cells : List ProjectEulerSolutions.MaskedStates.Cell)
    (view : ProjectEulerSolutions.MaskedStates.View
      (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage) base.length cells)
    (env : HeapCalls.Env) (getStates : env "new_states" = some (.ref base.length))
    (getMap : env "mp" = some (.ref dictionary)) (getDigits : env "digits" = some (.ref digitsAddress))
    (getUsed : env "used" = some (natural used)) :
    ∃ extra finalEnv finalCells,
      BlockContinues (globals := globals) program env stateBody
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)
        finalEnv (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (garbage ++ extra)) ∧
      ProjectEulerSolutions.MaskedStates.View
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (garbage ++ extra)) base.length finalCells ∧
      (∀ name, name ∉ ["a", "b", "c"] → finalEnv name = env name) ∧
      finalCells.map ProjectEulerSolutions.MaskedStates.Cell.state =
        cells.map ProjectEulerSolutions.MaskedStates.Cell.state ++
          collected (left :: right :: tail, used) (ProjectEulerSolutions.TupleBuckets.contents buckets (left, right)) := by
  let initial := ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage
  have original : initial[digitsAddress]? = some (.tuple (ProjectEulerSolutions.MaskedStates.numbers (left :: right :: tail))) :=
    ProjectEulerSolutions.HeapCollection.prefix_found base _ _ _ digitsFound
  have first : Evaluates (globals := globals) program env (.subscript (.name "digits") (.constant (.int 0)))
      initial (natural left) initial := by
    apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getDigits) Evaluates.constant
    simp [Mutable.getItem, original, Mutable.index, ProjectEulerSolutions.MaskedStates.numbers,
      ProjectEulerSolutions.MaskedStates.number, natural, Except.map]
  have second : Evaluates (globals := globals) program env (.subscript (.name "digits") (.constant (.int 1)))
      initial (natural right) initial := by
    apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getDigits) Evaluates.constant
    simp [Mutable.getItem, original, Mutable.index, ProjectEulerSolutions.MaskedStates.numbers,
      ProjectEulerSolutions.MaskedStates.number, natural, Except.map]
  let decoded := assignNames env ["b", "c"] [natural left, natural right]
  let pair := Mutable.Object.tuple [natural left, natural right]
  let decodedGarbage := garbage ++ [pair]
  let afterDecode := ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) decodedGarbage
  have decode : BlockContinues (globals := globals) program env
      [.assign [.tuple [.name "b", .name "c"]] pairExpression] initial decoded afterDecode := by
    apply BlockContinues.unpack (names := ["b", "c"]) (items := [natural left, natural right]) rfl
      (Evaluates.tuple (.cons first (.cons second .nil)) (by simp))
    · simp [Mutable.sequence, initial, afterDecode, decodedGarbage, ProjectEulerSolutions.HeapCollection.working, pair]
    · rfl
  have decodedView : ProjectEulerSolutions.MaskedStates.View afterDecode base.length cells := by
    simpa only [initial, pair, decodedGarbage, afterDecode, ProjectEulerSolutions.HeapCollection.append] using
      view.append [pair]
  have currentTable : ProjectEulerSolutions.TupleBuckets.View afterDecode dictionary buckets :=
    tableView.append (.list (ProjectEulerSolutions.MaskedStates.references cells) :: decodedGarbage)
  have lookup := ProjectEulerSolutions.TupleBuckets.evaluate_get (globals := globals) program decoded afterDecode dictionary buckets
    currentTable (left, right) (by simpa [decoded, assignNames, setLocal] using getMap)
      (by simp [decoded, assignNames, setLocal]; rfl) (by simp [decoded, assignNames, setLocal]; rfl)
  let prepared := [pair, Mutable.Object.list []]
  let lookupGarbage := decodedGarbage ++ prepared
  let afterLookup := ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) lookupGarbage
  have lookupHeap : ProjectEulerSolutions.TupleBuckets.prepared afterDecode (left, right) = afterLookup := by
    simp [ProjectEulerSolutions.TupleBuckets.prepared, ProjectEulerSolutions.TupleBuckets.number, natural,
      afterLookup, lookupGarbage, prepared, pair, afterDecode, ProjectEulerSolutions.HeapCollection.append]
  rw [lookupHeap] at lookup
  let sourceAddress := ProjectEulerSolutions.TupleBuckets.chosen buckets (left, right) (afterDecode.length+1)
  have sourceFound : afterLookup[sourceAddress]? = some (.list (ProjectEulerSolutions.MaskedStates.numbers
      (ProjectEulerSolutions.TupleBuckets.contents buckets (left, right)))) := by
    rw [← lookupHeap]
    exact ProjectEulerSolutions.TupleBuckets.get_values afterDecode dictionary buckets currentTable (left, right)
  have sourceDifferent : base.length ≠ sourceAddress := by
    cases selected : ProjectEulerSolutions.TupleBuckets.find buckets (left, right) with
    | none =>
        simp only [sourceAddress, ProjectEulerSolutions.TupleBuckets.chosen, selected, Option.map_none, Option.getD_none]
        simp only [afterDecode, ProjectEulerSolutions.HeapCollection.working, List.length_append, List.length_cons]
        omega
    | some bucket =>
        have inside := (List.getElem?_eq_some_iff.mp
          (tableView.lists bucket (List.mem_of_find?_eq_some selected))).1
        simpa only [sourceAddress, ProjectEulerSolutions.TupleBuckets.chosen, selected, Option.map_some, Option.getD_some] using
          (show base.length ≠ bucket.address by omega)
  have digitsDifferent : base.length ≠ digitsAddress := by
    have inside := (List.getElem?_eq_some_iff.mp digitsFound).1
    omega
  have afterView : ProjectEulerSolutions.MaskedStates.View afterLookup base.length cells := by
    simpa only [afterLookup, lookupGarbage, afterDecode, ProjectEulerSolutions.HeapCollection.append] using decodedView.append prepared
  have afterDigits : afterLookup[digitsAddress]? = some (.tuple (ProjectEulerSolutions.MaskedStates.numbers (left :: right :: tail))) :=
    ProjectEulerSolutions.HeapCollection.prefix_found base _ _ _ digitsFound
  obtain ⟨extra, finalEnv, finalCells, executed, finalView, untouched, values⟩ :=
    candidates_result (globals := globals) (ProjectEulerSolutions.TupleBuckets.contents buckets (left, right)) base
      sourceAddress sourceDifferent (left :: right :: tail, used) (by simp) digitsAddress digitsDifferent
      _ 0 rfl lookupGarbage cells afterView sourceFound afterDigits decoded
      (by simpa [decoded, assignNames, setLocal] using getStates)
      (by simpa [decoded, assignNames, setLocal] using getDigits) (by simpa [decoded, assignNames, setLocal] using getUsed)
  have loop : BlockContinues (globals := globals) program decoded
      [.forIn (.name "a") ProjectEulerSolutions.TupleBuckets.lookupExpression candidateBody []]
      afterDecode finalEnv
      (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (lookupGarbage ++ extra)) :=
    BlockContinues.for_sequence (value := .ref sourceAddress)
      (items := ProjectEulerSolutions.MaskedStates.numbers (ProjectEulerSolutions.TupleBuckets.contents buckets (left, right)))
      (by intros; simp [ProjectEulerSolutions.TupleBuckets.lookupExpression]) lookup
      (by have sequenceList (heap : Mutable.Heap) (address : Nat) (items : List Mutable.Value)
              (found : heap[address]? = some (.list items)) : Mutable.sequence (.ref address) heap = some (.ok items, heap) := by
            simp [Mutable.sequence, found]
          exact sequenceList _ _ _ sourceFound) executed
  refine ⟨[pair] ++ prepared ++ extra, finalEnv, finalCells, ?_, ?_, ?_, values⟩
  · simpa only [lookupGarbage, decodedGarbage, List.append_assoc] using decode.append loop
  · simpa only [lookupGarbage, decodedGarbage, List.append_assoc] using finalView
  · intro name unrelated
    have na : name ≠ "a" := by intro equal; subst name; simp at unrelated
    have nb : name ≠ "b" := by intro equal; subst name; simp at unrelated
    have nc : name ≠ "c" := by intro equal; subst name; simp at unrelated
    rw [untouched name na]
    simp [decoded, assignNames, setLocal, nb, nc]

private def generated (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (cells : List ProjectEulerSolutions.MaskedStates.Cell) : List ProjectEulerSolutions.MaskedStates.State :=
  cells.flatMap fun cell => collected cell.state
    (ProjectEulerSolutions.TupleBuckets.contents buckets (cell.state.1.headD 0, (cell.state.1.drop 1).headD 0))

/-- Traverse the original state list while collecting fresh states in a distinct list. -/
theorem states_result {globals : Mutable.Globals} (base : Mutable.Heap)
    (dictionary : Nat) (buckets : List ProjectEulerSolutions.TupleBuckets.Bucket)
    (tableView : ProjectEulerSolutions.TupleBuckets.View base dictionary buckets)
    (all : List ProjectEulerSolutions.MaskedStates.Cell) (statesAddress : Nat)
    (statesFound : base[statesAddress]? = some (.list (ProjectEulerSolutions.MaskedStates.references all)))
    (stored : ∀ cell ∈ all, ProjectEulerSolutions.MaskedStates.Stored base cell)
    (long : ∀ cell ∈ all, 2 ≤ cell.state.1.length)
    (remaining : List ProjectEulerSolutions.MaskedStates.Cell) (position : Nat) (split : all.drop position = remaining)
    (garbage : Mutable.Heap) (cells : List ProjectEulerSolutions.MaskedStates.Cell)
    (view : ProjectEulerSolutions.MaskedStates.View
      (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage) base.length cells)
    (env : HeapCalls.Env) (getStates : env "new_states" = some (.ref base.length))
    (getMap : env "mp" = some (.ref dictionary)) :
    ∃ extra finalEnv finalCells,
      BlockContinues (globals := globals) program env
        [.unpackStepLoop ["digits", "used"] (.ref statesAddress) position stateBody []]
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage)
        finalEnv (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (garbage ++ extra)) ∧
      ProjectEulerSolutions.MaskedStates.View
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references finalCells) (garbage ++ extra)) base.length finalCells ∧
      (∀ name, name ∉ ["digits", "used", "a", "b", "c"] → finalEnv name = env name) ∧
      finalCells.map ProjectEulerSolutions.MaskedStates.Cell.state =
        cells.map ProjectEulerSolutions.MaskedStates.Cell.state ++ generated buckets remaining := by
  have sequence (cells : List ProjectEulerSolutions.MaskedStates.Cell) (garbage : Mutable.Heap) :
      Mutable.sequence (.ref statesAddress)
        (ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage) =
      some (.ok (ProjectEulerSolutions.MaskedStates.references all),
        ProjectEulerSolutions.HeapCollection.working base (ProjectEulerSolutions.MaskedStates.references cells) garbage) := by
    have found := ProjectEulerSolutions.HeapCollection.prefix_found base
      (.list (ProjectEulerSolutions.MaskedStates.references cells) :: garbage) _ _ statesFound
    simp [Mutable.sequence, ProjectEulerSolutions.HeapCollection.working, found]
  induction remaining generalizing position garbage cells env with
  | nil =>
      refine ⟨[], env, cells, ?_, by simpa using view, ?_, by simp [generated]⟩
      · simp only [List.append_nil]
        exact BlockContinues.unpack_done (sequence cells garbage)
          (by have empty := congrArg List.head? split
              simpa [ProjectEulerSolutions.MaskedStates.references, List.head?_drop] using empty) .nil
      · intros; rfl
  | cons old rest ih =>
      have member : old ∈ all := List.mem_of_mem_drop (by rw [split]; exact List.mem_cons_self)
      have length := long old member
      obtain ⟨left, right, tail, shape⟩ : ∃ left right tail, old.state.1 = left :: right :: tail := by
        cases suffix : old.state.1 with
        | nil => simp [suffix] at length
        | cons left rest =>
            cases rest with
            | nil => simp [suffix] at length
            | cons right tail => exact ⟨left, right, tail, rfl⟩
      let bound := assignNames env ["digits", "used"] [.ref old.digitsAddress, natural old.state.2]
      obtain ⟨extra, middleEnv, nextCells, iteration, nextView, same, nextValues⟩ :=
        row (globals := globals) left right old.state.2 tail base dictionary buckets tableView
          old.digitsAddress (by simpa only [shape] using (stored old member).1) garbage cells view bound
          (by simpa [bound, assignNames, setLocal] using getStates)
          (by simpa [bound, assignNames, setLocal] using getMap)
          (by simp [bound, assignNames, setLocal]) (by simp [bound, assignNames, setLocal])
      obtain ⟨tailExtra, finalEnv, finalCells, executed, finalView, untouched, values⟩ :=
        ih (position+1) (by simp [List.drop_add_one_eq_tail_drop, split]) (garbage ++ extra) nextCells nextView middleEnv
          (by rw [same "new_states" (by decide)]; simpa [bound, assignNames, setLocal] using getStates)
          (by rw [same "mp" (by decide)]; simpa [bound, assignNames, setLocal] using getMap)
      refine ⟨extra ++ tailExtra, finalEnv, finalCells, ?_, by simpa only [List.append_assoc] using finalView, ?_, ?_⟩
      · have selected : all[position]? = some old := by
          simpa only [split, List.head?_cons] using (List.head?_drop (l := all) (i := position)).symm
        apply BlockContinues.unpack_step (source := .ref statesAddress) (items := ProjectEulerSolutions.MaskedStates.references all)
          (value := .ref old.address) (values := [.ref old.digitsAddress, natural old.state.2])
          (sequence cells garbage) (by simp [ProjectEulerSolutions.MaskedStates.references, ProjectEulerSolutions.MaskedStates.reference, selected])
          (by have found := ProjectEulerSolutions.HeapCollection.prefix_found base
                (.list (ProjectEulerSolutions.MaskedStates.references cells) :: garbage) _ _ (stored old member).2
              simp [Mutable.sequence, ProjectEulerSolutions.HeapCollection.working, found]; rfl) rfl
        simpa only [Calls.loopContinuation, show Calls.loopControl stateBody = false from rfl,
          Bool.false_eq_true, if_false, List.append_assoc] using iteration.append executed
      · intro name unrelated
        have nd : name ≠ "digits" := by intro equal; subst name; simp at unrelated
        have nu : name ≠ "used" := by intro equal; subst name; simp at unrelated
        have others : name ∉ ["a", "b", "c"] := fun inside =>
          unrelated (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ inside))
        rw [untouched name unrelated, same name others]
        simp [bound, assignNames, setLocal, nd, nu]
      · have stateEqual : (left :: right :: tail, old.state.2) = old.state := Prod.ext shape.symm rfl
        rw [values, nextValues, stateEqual]
        simp only [generated, List.flatMap_cons, shape, List.headD_cons, List.drop_succ_cons, List.drop_zero]
        exact List.append_assoc _ _ _

def primeBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3 with
  | .forIn _ _ body _ :: _ => body
  | _ => []

def sourceSetup : List Stmt := primeBody.take 2

def sourceStatesLoop : List Stmt := (primeBody.drop 2).take 1

/-- One prime pass evaluates the table lookup, allocates a new collector and rebinds the state list. -/
theorem prime_result {globals : Mutable.Globals} (p : Nat) (heap : Mutable.Heap)
    (mapsAddress : Nat) (entries : List BuildMaps.PrimeTable) (mapsView : BuildMaps.MapsView heap mapsAddress entries)
    (present : p ∈ entries.map BuildMaps.PrimeTable.modulus)
    (old : List ProjectEulerSolutions.MaskedStates.Cell) (statesAddress : Nat)
    (statesFound : heap[statesAddress]? = some (.list (ProjectEulerSolutions.MaskedStates.references old)))
    (stored : ∀ cell ∈ old, ProjectEulerSolutions.MaskedStates.Stored heap cell)
    (long : ∀ cell ∈ old, 2 ≤ cell.state.1.length)
    (env : HeapCalls.Env) (getMaps : env "maps" = some (.ref mapsAddress))
    (getStates : env "states" = some (.ref statesAddress)) (getP : env "p" = some (natural p)) :
    ∃ garbage finalEnv cells,
      BlockContinues (globals := globals) program env primeBody heap finalEnv
        (ProjectEulerSolutions.HeapCollection.working heap (ProjectEulerSolutions.MaskedStates.references cells) garbage) ∧
      ProjectEulerSolutions.MaskedStates.View
        (ProjectEulerSolutions.HeapCollection.working heap (ProjectEulerSolutions.MaskedStates.references cells) garbage) heap.length cells ∧
      finalEnv "states" = some (.ref heap.length) ∧
      (∀ name, name ∉ ["mp", "new_states", "digits", "used", "a", "b", "c", "states"] → finalEnv name = env name) ∧
      cells.map ProjectEulerSolutions.MaskedStates.Cell.state = extend p (old.map ProjectEulerSolutions.MaskedStates.Cell.state) := by
  obtain ⟨dictionary, buckets, lookup, tableView, correct⟩ := BuildMaps.table_lookup heap mapsAddress entries mapsView p present
  let mapped := setLocal env "mp" (.ref dictionary)
  let ready := setLocal mapped "new_states" (.ref heap.length)
  have setup : BlockContinues (globals := globals) program env sourceSetup heap ready (heap ++ [.list []]) := by
    exact (BlockContinues.assign (Evaluates.subscript (by intros; simp) (Evaluates.lookup getMaps) (Evaluates.lookup getP) lookup)).append
      (BlockContinues.annAssign (Evaluates.list .nil))
  have initialized : ProjectEulerSolutions.MaskedStates.View
      (ProjectEulerSolutions.HeapCollection.working heap (ProjectEulerSolutions.MaskedStates.references []) []) heap.length [] := by
    constructor <;> simp
  obtain ⟨garbage, middleEnv, cells, executed, finalView, untouched, values⟩ :=
    states_result (globals := globals) heap dictionary buckets tableView old statesAddress statesFound stored long
      old 0 rfl [] [] initialized ready (by simp [ready, setLocal]) (by simp [ready, mapped, setLocal])
  have loop : BlockContinues (globals := globals) program ready sourceStatesLoop (heap ++ [.list []]) middleEnv
      (ProjectEulerSolutions.HeapCollection.working heap (ProjectEulerSolutions.MaskedStates.references cells) garbage) := by
    apply BlockContinues.for_unpack (value := .ref statesAddress) (items := ProjectEulerSolutions.MaskedStates.references old)
      (by intros; simp) rfl (Evaluates.lookup (by simpa [ready, mapped, setLocal] using getStates))
      (by change Mutable.sequence (.ref statesAddress) (heap ++ [.list []]) = _
          have found := ProjectEulerSolutions.HeapCollection.prefix_found heap [.list []] _ _ statesFound
          simp [Mutable.sequence, found, ProjectEulerSolutions.HeapCollection.working, ProjectEulerSolutions.MaskedStates.references]) executed
  have getNew : middleEnv "new_states" = some (.ref heap.length) := by
    rw [untouched "new_states" (by decide)]
    simp [ready, setLocal]
  let finalEnv := setLocal middleEnv "states" (.ref heap.length)
  refine ⟨garbage, finalEnv, cells, ?_, finalView, by simp [finalEnv, setLocal], ?_, ?_⟩
  · exact (setup.append loop).append (BlockContinues.assign (Evaluates.lookup getNew))
  · intro name unrelated
    have ns : name ≠ "states" := by intro equal; subst name; simp at unrelated
    have nn : name ≠ "new_states" := by intro equal; subst name; simp at unrelated
    have nm : name ≠ "mp" := by intro equal; subst name; simp at unrelated
    have others : name ∉ ["digits", "used", "a", "b", "c"] := by
      intro inside
      simp only [List.mem_cons, List.not_mem_nil, or_false] at inside
      rcases inside with a | b | c | d | e <;> subst name <;> simp at unrelated
    simp only [finalEnv, setLocal, if_neg ns]
    rw [untouched name others]
    simp [ready, mapped, setLocal, nn, nm]
  · simpa only [List.map_nil, List.nil_append, generated, correct, extend, collected, List.flatMap_map] using values

private theorem extend_long (p : Nat) (previous : List ProjectEulerSolutions.MaskedStates.State)
    (long : ∀ state ∈ previous, 2 ≤ state.1.length) :
    ∀ state ∈ extend p previous, 2 ≤ state.1.length := by
  intro state member
  obtain ⟨old, inside, next⟩ := List.mem_flatMap.mp member
  obtain ⟨digit, _, rfl⟩ := List.mem_map.mp next
  have length := long old inside
  simp only [List.length_cons]
  omega

/-- The six prime passes retain the original heap and compute the left fold of the source transition. -/
theorem outer_result {globals : Mutable.Globals} (all : List Nat) (base : Mutable.Heap) (primeAddress : Nat)
    (foundPrimes : base[primeAddress]? = some (.list (all.map natural)))
    (mapsAddress : Nat) (entries : List BuildMaps.PrimeTable)
    (remaining : List Nat) (position : Nat) (split : all.drop position = remaining)
    (present : ∀ p ∈ remaining, p ∈ entries.map BuildMaps.PrimeTable.modulus)
    (suffix : Mutable.Heap) (mapsView : BuildMaps.MapsView (base ++ suffix) mapsAddress entries)
    (old : List ProjectEulerSolutions.MaskedStates.Cell) (statesAddress : Nat)
    (statesFound : (base ++ suffix)[statesAddress]? = some (.list (ProjectEulerSolutions.MaskedStates.references old)))
    (stored : ∀ cell ∈ old, ProjectEulerSolutions.MaskedStates.Stored (base ++ suffix) cell)
    (long : ∀ cell ∈ old, 2 ≤ cell.state.1.length)
    (env : HeapCalls.Env) (getMaps : env "maps" = some (.ref mapsAddress))
    (getStates : env "states" = some (.ref statesAddress)) :
    ∃ extra finalEnv cells finalAddress,
      BlockContinues (globals := globals) program env [.sequenceStepLoop "p" (.ref primeAddress) position primeBody []]
        (base ++ suffix) finalEnv (base ++ extra) ∧
      BuildMaps.MapsView (base ++ extra) mapsAddress entries ∧
      finalEnv "states" = some (.ref finalAddress) ∧
      (base ++ extra)[finalAddress]? = some (.list (ProjectEulerSolutions.MaskedStates.references cells)) ∧
      (∀ cell ∈ cells, ProjectEulerSolutions.MaskedStates.Stored (base ++ extra) cell) ∧
      (∀ cell ∈ cells, 2 ≤ cell.state.1.length) ∧
      (∀ name, name ∉ ["p", "mp", "new_states", "digits", "used", "a", "b", "c", "states"] → finalEnv name = env name) ∧
      cells.map ProjectEulerSolutions.MaskedStates.Cell.state =
        remaining.foldl (fun previous p => extend p previous) (old.map ProjectEulerSolutions.MaskedStates.Cell.state) := by
  have sequence (extra : Mutable.Heap) : Mutable.sequence (.ref primeAddress) (base ++ extra) =
      some (.ok (all.map natural), base ++ extra) := by
    have found := ProjectEulerSolutions.HeapCollection.prefix_found base extra _ _ foundPrimes
    simp [Mutable.sequence, found]
  induction remaining generalizing position suffix old statesAddress env with
  | nil =>
      refine ⟨suffix, env, old, statesAddress, ?_, mapsView, getStates, statesFound, stored, long, ?_, rfl⟩
      · exact BlockContinues.sequence_done (sequence suffix)
          (by have empty := congrArg List.head? split; simpa [List.head?_drop] using empty) .nil
      · intros; rfl
  | cons p rest ih =>
      let bound := setLocal env "p" (natural p)
      obtain ⟨garbage, middleEnv, nextCells, iteration, nextView, nextStates, same, nextValues⟩ :=
        prime_result (globals := globals) p (base ++ suffix) mapsAddress entries mapsView (present p List.mem_cons_self)
          old statesAddress statesFound stored long bound
          (by simpa [bound, setLocal] using getMaps) (by simpa [bound, setLocal] using getStates) (by simp [bound, setLocal])
      let nextSuffix := suffix ++ (.list (ProjectEulerSolutions.MaskedStates.references nextCells) :: garbage)
      have nextHeap : ProjectEulerSolutions.HeapCollection.working (base ++ suffix)
          (ProjectEulerSolutions.MaskedStates.references nextCells) garbage = base ++ nextSuffix := by
        simp [ProjectEulerSolutions.HeapCollection.working, nextSuffix, List.append_assoc]
      have nextMapsView : BuildMaps.MapsView (base ++ nextSuffix) mapsAddress entries := by
        simpa only [nextSuffix, List.append_assoc] using mapsView.append
          (.list (ProjectEulerSolutions.MaskedStates.references nextCells) :: garbage)
      have nextFound : (base ++ nextSuffix)[(base ++ suffix).length]? = some (.list (ProjectEulerSolutions.MaskedStates.references nextCells)) := by
        rw [← nextHeap]
        exact ProjectEulerSolutions.HeapCollection.found _ _ _
      have nextStored : ∀ cell ∈ nextCells, ProjectEulerSolutions.MaskedStates.Stored (base ++ nextSuffix) cell := by
        rw [← nextHeap]
        exact nextView.stored
      have nextLong : ∀ cell ∈ nextCells, 2 ≤ cell.state.1.length := by
        intro cell member
        have inside := List.mem_map_of_mem (f := ProjectEulerSolutions.MaskedStates.Cell.state) member
        rw [nextValues] at inside
        exact extend_long p _ (by intro state h; obtain ⟨oldCell, oldMember, rfl⟩ := List.mem_map.mp h; exact long oldCell oldMember) _ inside
      rw [nextHeap] at iteration
      obtain ⟨extra, finalEnv, cells, finalAddress, executed, finalMaps, finalStates, finalFound, finalStored, finalLong, untouched, values⟩ :=
        ih (position+1) (by simp [List.drop_add_one_eq_tail_drop, split])
          (fun q member => present q (List.mem_cons_of_mem _ member)) nextSuffix nextMapsView nextCells
          (base ++ suffix).length nextFound nextStored nextLong middleEnv
          (by rw [same "maps" (by decide)]; simpa [bound, setLocal] using getMaps) nextStates
      refine ⟨extra, finalEnv, cells, finalAddress, ?_, finalMaps, finalStates, finalFound, finalStored, finalLong, ?_, ?_⟩
      · have selected : all[position]? = some p := by
          simpa only [split, List.head?_cons] using (List.head?_drop (l := all) (i := position)).symm
        apply BlockContinues.sequence_step (source := .ref primeAddress) (value := natural p)
          (sequence suffix) (by simp [selected])
        simpa only [Calls.loopContinuation, show Calls.loopControl primeBody = false from rfl,
          Bool.false_eq_true, if_false] using iteration.append executed
      · intro name unrelated
        have np : name ≠ "p" := by intro equal; subst name; simp at unrelated
        have others : name ∉ ["mp", "new_states", "digits", "used", "a", "b", "c", "states"] :=
          fun member => unrelated (List.mem_cons_of_mem _ member)
        rw [untouched name unrelated, same name others]
        simp [bound, setLocal, np]
      · simpa only [List.foldl_cons, nextValues] using values

private def primes : List Nat := [13, 11, 7, 5, 3, 2]
def sourceLoop : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3).take 1

/-- Execute all source extensions, including the once-allocated prime list and every intermediate state list. -/
theorem result {globals : Mutable.Globals} (heap : Mutable.Heap)
    (mapsAddress : Nat) (entries : List BuildMaps.PrimeTable) (mapsView : BuildMaps.MapsView heap mapsAddress entries)
    (present : ∀ p ∈ primes, p ∈ entries.map BuildMaps.PrimeTable.modulus)
    (old : List ProjectEulerSolutions.MaskedStates.Cell) (statesAddress : Nat)
    (statesFound : heap[statesAddress]? = some (.list (ProjectEulerSolutions.MaskedStates.references old)))
    (stored : ∀ cell ∈ old, ProjectEulerSolutions.MaskedStates.Stored heap cell)
    (long : ∀ cell ∈ old, 2 ≤ cell.state.1.length)
    (env : HeapCalls.Env) (getMaps : env "maps" = some (.ref mapsAddress))
    (getStates : env "states" = some (.ref statesAddress)) :
    ∃ extra finalEnv cells finalAddress,
      BlockContinues (globals := globals) program env sourceLoop heap finalEnv (heap ++ extra) ∧
      finalEnv "states" = some (.ref finalAddress) ∧
      (heap ++ extra)[finalAddress]? = some (.list (ProjectEulerSolutions.MaskedStates.references cells)) ∧
      (∀ cell ∈ cells, ProjectEulerSolutions.MaskedStates.Stored (heap ++ extra) cell) ∧
      (∀ name, name ∉ ["p", "mp", "new_states", "digits", "used", "a", "b", "c", "states"] → finalEnv name = env name) ∧
      cells.map ProjectEulerSolutions.MaskedStates.Cell.state =
        primes.foldl (fun previous p => extend p previous) (old.map ProjectEulerSolutions.MaskedStates.Cell.state) := by
  let literal := Mutable.Object.list (primes.map natural)
  let base := heap ++ [literal]
  have extendedMaps := mapsView.append [literal]
  have extendedStates := ProjectEulerSolutions.HeapCollection.prefix_found heap [literal] _ _ statesFound
  obtain ⟨extra, finalEnv, cells, finalAddress, executed, _, finalStates, finalFound, finalStored, _, untouched, values⟩ :=
    outer_result (globals := globals) primes base heap.length (by simp [base, literal]) mapsAddress entries primes 0 rfl present
      [] (by simpa [base] using extendedMaps) old statesAddress (by simpa using extendedStates)
      (by intro cell member; simpa using (stored cell member).append [literal]) long env getMaps getStates
  refine ⟨[literal] ++ extra, finalEnv, cells, finalAddress, ?_, finalStates, ?_, ?_, untouched, values⟩
  · apply BlockContinues.for_sequence (value := .ref heap.length) (items := primes.map natural) (by intros; simp)
      (Evaluates.list (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant
        (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))))))
      (by simp [Mutable.sequence, primes, natural])
    simpa only [base, List.append_nil, List.append_assoc] using executed
  · simpa only [base, List.append_assoc] using finalFound
  · simpa only [base, List.append_assoc] using finalStored

end ExtendStates

namespace CompleteStates
open ProjectEulerSolutions.SubstringSearch ProjectEulerSolutions.SubstringDivisibility
private abbrev natural (n : Nat) : Mutable.Value := .scalar (.int n)
private def appendDigit : List Stmt := [.assign [.name "num"]
  (.binOp (.binOp (.name "num") .mult (.constant (.int 10))) .add (.name "d"))]
private def decimal (initial : Nat) (suffix : List Nat) : Nat :=
  suffix.foldl (fun n d => n*10+d) initial

private theorem decimal_value (initial : Nat) (suffix : List Nat) :
    decimal initial suffix = initial * 10^suffix.length + value suffix := by
  induction suffix generalizing initial with
  | nil => simp [decimal, value]
  | cons digit rest ih =>
      change decimal (initial*10+digit) rest = _
      rw [ih]
      simp only [value, Nat.ofDigits_reverse_cons, List.length_cons, pow_succ]
      ring

private theorem decimal_cons (initial : Nat) (suffix : List Nat) :
    decimal initial suffix = value (initial :: suffix) := by
  rw [decimal_value]
  simp only [value, Nat.ofDigits_reverse_cons]
  ring

set_option maxRecDepth 4000 in
set_option maxHeartbeats 2000000 in
/-- Every reachable final suffix has one missing nonzero digit; its bit length agrees with the mask model. -/
theorem reachable : ∀ state ∈ states, let remaining := 1023 ^^^ state.2
    0 < remaining ∧ remaining &&& (remaining-1) = 0 ∧ 1 < remaining.size ∧
      FormalSW.Py.Builtin.Int.bitLength remaining = remaining.size := by
  decide +kernel

/-- The decimal assembly loop reads the original suffix tuple and changes only its two loop locals. -/
theorem digits_result {globals : Mutable.Globals} (all : List Nat) (address : Nat) (heap : Mutable.Heap)
    (found : heap[address]? = some (.tuple (ProjectEulerSolutions.MaskedStates.numbers all)))
    (remaining : List Nat) (position : Nat) (split : all.drop position = remaining)
    (initial : Nat) (env : HeapCalls.Env) (getNum : env "num" = some (natural initial)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env [.sequenceStepLoop "d" (.ref address) position appendDigit []]
        heap finalEnv heap ∧ finalEnv "num" = some (natural (decimal initial remaining)) ∧
      (∀ name, name ∉ ["d", "num"] → finalEnv name = env name) := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (ProjectEulerSolutions.MaskedStates.numbers all), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position initial env with
  | nil =>
      refine ⟨env, ?_, getNum, ?_⟩
      · exact BlockContinues.sequence_done sequence
          (by have empty := congrArg List.head? split
              simpa [ProjectEulerSolutions.MaskedStates.numbers, List.head?_drop] using empty) .nil
      · intros; rfl
  | cons digit rest ih =>
      let bound := setLocal env "d" (natural digit)
      let ready := setLocal bound "num" (natural (initial*10+digit))
      have step : BlockContinues (globals := globals) program bound appendDigit heap ready heap := by
        apply BlockContinues.assign
        apply Evaluates.binary (av := natural (initial*10)) (bv := natural digit)
          (middle := heap) (afterRight := heap) (finalHeap := heap)
          (Evaluates.binary (av := natural initial) (bv := natural 10) (middle := heap) (afterRight := heap) (finalHeap := heap)
            (Evaluates.lookup (by simpa [bound, setLocal] using getNum)) Evaluates.constant
            (by simp [Mutable.binary, scalarBinary, natural]))
          (Evaluates.lookup (by simp [bound, setLocal]))
        simp [Mutable.binary, scalarBinary, natural]
      obtain ⟨finalEnv, executed, finalNum, untouched⟩ :=
        ih (position+1) (by simp [List.drop_add_one_eq_tail_drop, split]) (initial*10+digit) ready (by simp [ready, setLocal])
      refine ⟨finalEnv, ?_, finalNum, ?_⟩
      · have selected : all[position]? = some digit := by
          simpa only [split, List.head?_cons] using (List.head?_drop (l := all) (i := position)).symm
        apply BlockContinues.sequence_step (source := .ref address) (value := natural digit) sequence
          (by simp [ProjectEulerSolutions.MaskedStates.numbers, selected]; rfl)
        simpa only [Calls.loopContinuation, show Calls.loopControl appendDigit = false from rfl,
          Bool.false_eq_true, if_false] using step.append executed
      · intro name unrelated
        have nd : name ≠ "d" := by intro equal; subst name; simp at unrelated
        have nn : name ≠ "num" := by intro equal; subst name; simp at unrelated
        rw [untouched name unrelated]
        simp [ready, bound, setLocal, nd, nn]

private def remainingExpression : Expr := .binOp (.name "all_mask") .bitXor (.name "used")
private def rejectMask : Expr := .boolOp .or
  [.compare (.name "remaining_mask") [(.eq, .constant (.int 0))],
   .compare (.binOp (.name "remaining_mask") .bitAnd (.binOp (.name "remaining_mask") .sub (.constant (.int 1))))
     [(.notEq, .constant (.int 0))]]
private def firstExpression : Expr := .binOp (.builtinCall .intBitLength [.name "remaining_mask"]) .sub (.constant (.int 1))
private def rejectFirst : Expr := .compare (.name "d1") [(.eq, .constant (.int 0))]
def body : List Stmt :=
  [.assign [.name "remaining_mask"] remainingExpression, .ifThen rejectMask [.continueLoop] [],
   .assign [.name "d1"] firstExpression, .ifThen rejectFirst [.continueLoop] [],
   .assign [.name "num"] (.name "d1"), .forIn (.name "d") (.name "suffix") appendDigit [],
   .augAssign (.name "total") .add (.name "num")]
def sourceBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 6 with
  | .forIn _ _ body _ :: _ => body
  | _ => []

theorem body_source : body = sourceBody := rfl

/-- Complete one reachable suffix using the actual bit-length call, decimal loop and total update. -/
theorem row {globals : Mutable.Globals} (state : ProjectEulerSolutions.SubstringSearch.State) (member : state ∈ states)
    (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.tuple (ProjectEulerSolutions.MaskedStates.numbers state.1)))
    (initial : Nat) (env : HeapCalls.Env) (getTotal : env "total" = some (natural initial))
    (getAll : env "all_mask" = some (natural 1023)) (getUsed : env "used" = some (natural state.2))
    (getSuffix : env "suffix" = some (.ref address)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env body heap finalEnv heap ∧
      finalEnv "total" = some (natural (initial + ((complete state).map value).sum)) ∧
      (∀ name, name ∉ ["remaining_mask", "d1", "num", "d", "total"] → finalEnv name = env name) := by
  let remaining := 1023 ^^^ state.2
  obtain ⟨positive, single, size, bits⟩ := reachable state member
  change 0 < remaining at positive
  change remaining &&& (remaining-1) = 0 at single
  change 1 < remaining.size at size
  change FormalSW.Py.Builtin.Int.bitLength remaining = remaining.size at bits
  have nonzero : remaining ≠ 0 := by omega
  have atLeastOne : 1 ≤ remaining := by omega
  let first := remaining.size-1
  have firstNonzero : first ≠ 0 := by dsimp [first]; omega
  have completed : complete state = [first :: state.1] := by
    simp only [complete, ← show remaining = 1023 ^^^ state.2 from rfl, nonzero, single, ne_eq,
      not_true_eq_false, or_self, if_false, ← show first = remaining.size-1 from rfl, firstNonzero]
  let masked := setLocal env "remaining_mask" (natural remaining)
  have setMask : BlockContinues (globals := globals) program env [.assign [.name "remaining_mask"] remainingExpression] heap masked heap :=
    BlockContinues.assign (Evaluates.binary (Evaluates.lookup getAll) (Evaluates.lookup getUsed)
      (by simp [Mutable.binary, scalarBitXor, scalarNumber, natural, remaining]; rfl))
  have getRemaining : masked "remaining_mask" = some (natural remaining) := by simp [masked, setLocal]
  have zeroTest : Evaluates (globals := globals) program masked (.compare (.name "remaining_mask") [(.eq, .constant (.int 0))])
      heap (.scalar (.bool false)) heap :=
    Evaluates.compare (Evaluates.lookup getRemaining) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonzero])
  have minus : Evaluates (globals := globals) program masked (.binOp (.name "remaining_mask") .sub (.constant (.int 1)))
      heap (natural (remaining-1)) heap :=
    Evaluates.binary (Evaluates.lookup getRemaining) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, natural, Int.natCast_sub atLeastOne])
  have andResult : Evaluates (globals := globals) program masked
      (.binOp (.name "remaining_mask") .bitAnd (.binOp (.name "remaining_mask") .sub (.constant (.int 1))))
      heap (natural 0) heap := by
    simpa only [single] using Evaluates.binary (Evaluates.lookup getRemaining) minus
      (show Mutable.binary .bitAnd (natural remaining) (natural (remaining-1)) heap =
        some (.ok (natural (remaining &&& (remaining-1))), heap) by simp [Mutable.binary, scalarBitAnd, scalarNumber, natural])
  have singleTest : Evaluates (globals := globals) program masked
      (.compare (.binOp (.name "remaining_mask") .bitAnd (.binOp (.name "remaining_mask") .sub (.constant (.int 1))))
        [(.notEq, .constant (.int 0))]) heap (.scalar (.bool false)) heap :=
    Evaluates.compare andResult Evaluates.constant rfl
  have tested : Evaluates (globals := globals) program masked rejectMask heap (.scalar (.bool false)) heap :=
    Evaluates.boolean_next (flag := false) zeroTest rfl rfl (Evaluates.boolean_last singleTest rfl)
  have keepMask : BlockContinues (globals := globals) program masked [.ifThen rejectMask [.continueLoop] []] heap masked heap :=
    BlockContinues.branch tested rfl .nil
  have bitLength : Evaluates (globals := globals) program masked (.builtinCall .intBitLength [.name "remaining_mask"])
      heap (natural remaining.size) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getRemaining) .nil)
    change scalarResult (some (Math.bitLengthCall (.int remaining))) heap = _
    simp only [Math.bitLengthCall, scalarNumber]
    rw [bits]
    rfl
  let headed := setLocal masked "d1" (natural first)
  have setFirst : BlockContinues (globals := globals) program masked [.assign [.name "d1"] firstExpression] heap headed heap :=
    BlockContinues.assign (Evaluates.binary bitLength Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, natural, first, Int.natCast_sub (show 1 ≤ remaining.size by omega)]))
  have firstTest : Evaluates (globals := globals) program headed rejectFirst heap (.scalar (.bool false)) heap :=
    Evaluates.compare (av := natural first) (bv := natural 0) (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup (by simp [headed, setLocal])) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, firstNonzero])
  have keepFirst : BlockContinues (globals := globals) program headed [.ifThen rejectFirst [.continueLoop] []] heap headed heap :=
    BlockContinues.branch firstTest rfl .nil
  let numbered := setLocal headed "num" (natural first)
  have setNum : BlockContinues (globals := globals) program headed [.assign [.name "num"] (.name "d1")] heap numbered heap :=
    BlockContinues.assign (Evaluates.lookup (by simp [headed, setLocal]))
  obtain ⟨middleEnv, executed, finalNum, untouched⟩ := digits_result (globals := globals) state.1 address heap found
    state.1 0 rfl first numbered (by simp [numbered, setLocal])
  have loop : BlockContinues (globals := globals) program numbered [.forIn (.name "d") (.name "suffix") appendDigit []]
      heap middleEnv heap :=
    BlockContinues.for_sequence (value := .ref address) (items := ProjectEulerSolutions.MaskedStates.numbers state.1)
      (by intros; simp) (Evaluates.lookup (by simpa [numbered, headed, masked, setLocal] using getSuffix))
      (by simp [Mutable.sequence, found]) executed
  have currentTotal : middleEnv "total" = some (natural initial) := by
    rw [untouched "total" (by decide)]
    simpa [numbered, headed, masked, setLocal] using getTotal
  rw [decimal_cons] at finalNum
  let finalEnv := setLocal middleEnv "total" (natural (initial + value (first :: state.1)))
  have addTotal : BlockContinues (globals := globals) program middleEnv [.augAssign (.name "total") .add (.name "num")]
      heap finalEnv heap :=
    BlockContinues.augAssign currentTotal (Evaluates.lookup finalNum)
      (by simp [inplace, Mutable.binary, scalarBinary, natural])
  refine ⟨finalEnv, ?_, ?_, ?_⟩
  · exact (((((setMask.append keepMask).append setFirst).append keepFirst).append setNum).append loop).append addTotal
  · simp [finalEnv, setLocal, completed]
  · intro name unrelated
    have nr : name ≠ "remaining_mask" := by intro equal; subst name; simp at unrelated
    have nf : name ≠ "d1" := by intro equal; subst name; simp at unrelated
    have nn : name ≠ "num" := by intro equal; subst name; simp at unrelated
    have nt : name ≠ "total" := by intro equal; subst name; simp at unrelated
    have others : name ∉ ["d", "num"] := by
      intro inside
      simp only [List.mem_cons, List.not_mem_nil, or_false] at inside
      rcases inside with a | b <;> subst name <;> simp at unrelated
    simp only [finalEnv, setLocal, if_neg nt]
    rw [untouched name others]
    simp [numbered, headed, masked, setLocal, nr, nf, nn]

private def contribution (state : ProjectEulerSolutions.SubstringSearch.State) : Nat := ((complete state).map value).sum
private def contributions (cells : List ProjectEulerSolutions.MaskedStates.Cell) : Nat :=
  (cells.map fun cell => contribution cell.state).sum

/-- The final source loop sums the completion of each reachable state without changing the heap. -/
theorem states_result {globals : Mutable.Globals} (all : List ProjectEulerSolutions.MaskedStates.Cell)
    (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ProjectEulerSolutions.MaskedStates.references all)))
    (stored : ∀ cell ∈ all, ProjectEulerSolutions.MaskedStates.Stored heap cell)
    (reachable : ∀ cell ∈ all, cell.state ∈ states)
    (remaining : List ProjectEulerSolutions.MaskedStates.Cell) (position : Nat) (split : all.drop position = remaining)
    (initial : Nat) (env : HeapCalls.Env) (getTotal : env "total" = some (natural initial))
    (getAll : env "all_mask" = some (natural 1023)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env [.unpackStepLoop ["suffix", "used"] (.ref address) position body []]
        heap finalEnv heap ∧ finalEnv "total" = some (natural (initial + contributions remaining)) ∧
      (∀ name, name ∉ ["suffix", "used", "remaining_mask", "d1", "num", "d", "total"] → finalEnv name = env name) := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (ProjectEulerSolutions.MaskedStates.references all), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position initial env with
  | nil =>
      refine ⟨env, ?_, by simpa [contributions] using getTotal, ?_⟩
      · exact BlockContinues.unpack_done sequence
          (by have empty := congrArg List.head? split
              simpa [ProjectEulerSolutions.MaskedStates.references, List.head?_drop] using empty) .nil
      · intros; rfl
  | cons cell rest ih =>
      have member : cell ∈ all := List.mem_of_mem_drop (by rw [split]; exact List.mem_cons_self)
      let bound := assignNames env ["suffix", "used"] [.ref cell.digitsAddress, natural cell.state.2]
      obtain ⟨middleEnv, iteration, nextTotal, same⟩ := row (globals := globals) cell.state (reachable cell member)
        heap cell.digitsAddress (stored cell member).1 initial bound
        (by simpa [bound, assignNames, setLocal] using getTotal)
        (by simpa [bound, assignNames, setLocal] using getAll)
        (by simp [bound, assignNames, setLocal]) (by simp [bound, assignNames, setLocal])
      obtain ⟨finalEnv, executed, finalTotal, untouched⟩ := ih (position+1)
        (by simp [List.drop_add_one_eq_tail_drop, split]) (initial + contribution cell.state) middleEnv nextTotal
        (by rw [same "all_mask" (by decide)]; simpa [bound, assignNames, setLocal] using getAll)
      refine ⟨finalEnv, ?_, ?_, ?_⟩
      · have selected : all[position]? = some cell := by
          simpa only [split, List.head?_cons] using (List.head?_drop (l := all) (i := position)).symm
        apply BlockContinues.unpack_step (source := .ref address) (items := ProjectEulerSolutions.MaskedStates.references all)
          (value := .ref cell.address) (values := [.ref cell.digitsAddress, natural cell.state.2]) sequence
          (by simp [ProjectEulerSolutions.MaskedStates.references, ProjectEulerSolutions.MaskedStates.reference, selected])
          (by simp [Mutable.sequence, (stored cell member).2]; rfl) rfl
        have followed := BuildMaps.Iterates.then_loop
          (BuildMaps.Iterates.prepend iteration (BuildMaps.Iterates.done _ _)) executed
        simpa only [Calls.loopContinuation, show Calls.loopControl body = true from rfl, if_true] using followed
      · simpa only [contributions, List.map_cons, List.sum_cons, Nat.add_assoc] using finalTotal
      · intro name unrelated
        have ns : name ≠ "suffix" := by intro equal; subst name; simp at unrelated
        have nu : name ≠ "used" := by intro equal; subst name; simp at unrelated
        have others : name ∉ ["remaining_mask", "d1", "num", "d", "total"] := fun inside =>
          unrelated (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ inside))
        rw [untouched name unrelated, same name others]
        simp [bound, assignNames, setLocal, ns, nu]

def sourceSetup : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 4).take 2
def sourceLoop : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 6).take 1

private theorem contributions_sum (cells : List ProjectEulerSolutions.MaskedStates.Cell) :
    contributions cells = ((cells.map ProjectEulerSolutions.MaskedStates.Cell.state).flatMap complete |>.map value).sum := by
  induction cells with
  | nil => rfl
  | cons cell rest ih =>
      simp only [contributions, List.map_cons, List.sum_cons, List.flatMap_cons, List.map_append,
        List.sum_append, contribution] at ih ⊢
      rw [ih]

/-- Source initialization and the complete final loop produce the mathematical search sum. -/
theorem result {globals : Mutable.Globals} (cells : List ProjectEulerSolutions.MaskedStates.Cell)
    (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ProjectEulerSolutions.MaskedStates.references cells)))
    (stored : ∀ cell ∈ cells, ProjectEulerSolutions.MaskedStates.Stored heap cell)
    (values : cells.map ProjectEulerSolutions.MaskedStates.Cell.state = states) (env : HeapCalls.Env)
    (getStates : env "states" = some (.ref address)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env (sourceSetup ++ sourceLoop) heap finalEnv heap ∧
      finalEnv "total" = some (natural ((ProjectEulerSolutions.SubstringSearch.rows.map value).sum)) := by
  let zeroed := setLocal env "total" (natural 0)
  let ready := setLocal zeroed "all_mask" (natural 1023)
  have setup : BlockContinues (globals := globals) program env sourceSetup heap ready heap := by
    exact (BlockContinues.assign Evaluates.constant).append
      (BlockContinues.assign (Evaluates.binary (Evaluates.binary Evaluates.constant Evaluates.constant rfl) Evaluates.constant rfl))
  have allReachable : ∀ cell ∈ cells, cell.state ∈ states := by
    intro cell member
    rw [← values]
    exact List.mem_map_of_mem (f := ProjectEulerSolutions.MaskedStates.Cell.state) member
  obtain ⟨finalEnv, executed, total, _⟩ := states_result (globals := globals) cells heap address found stored allReachable
    cells 0 rfl 0 ready (by simp [ready, zeroed, setLocal]) (by simp [ready, setLocal])
  have loop : BlockContinues (globals := globals) program ready sourceLoop heap finalEnv heap :=
    BlockContinues.for_unpack (value := .ref address) (items := ProjectEulerSolutions.MaskedStates.references cells)
      (by intros; simp) rfl (Evaluates.lookup (by simpa [ready, zeroed, setLocal] using getStates))
      (by simp [Mutable.sequence, found]) executed
  refine ⟨finalEnv, setup.append loop, ?_⟩
  simpa only [Nat.zero_add, contributions_sum, values, ProjectEulerSolutions.SubstringSearch.rows] using total

end CompleteStates

namespace solve
open ProjectEulerSolutions.SubstringSearch
private abbrev natural (n : Nat) : Mutable.Value := .scalar (.int n)

/-- The complete Python solver terminates, preserves its input heap and returns the mathematical search sum. -/
theorem result (heap : Mutable.Heap) (primeAddress : Nat)
    (foundPrimes : heap[primeAddress]? = some (.list (numbers Domain.initialized))) :
    ∃ extra, ReturnsValue (globals := Domain.bindings primeAddress) program "solve" [] heap
      (natural ((rows.map ProjectEulerSolutions.SubstringDivisibility.value).sum)) (heap ++ extra) := by
  let primes : List Nat := [2,3,5,7,11,13,17]
  obtain ⟨mapsExtra, entries, built, mapsView, coverage⟩ :=
    BuildMaps.result primes (by intro p member; simp [primes] at member; omega) heap primeAddress foundPrimes
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [] (Domain.bindings primeAddress)
  let mapped := setLocal initial "maps" (.ref heap.length)
  let mapHeap := heap ++ mapsExtra
  have setMaps : BlockContinues (globals := Domain.bindings primeAddress) program initial
      [.assign [.name "maps"] (.call (.name "build_maps") [])] heap mapped mapHeap :=
    BlockContinues.assign (Evaluates.call .nil built)
  obtain ⟨seedGarbage, seededEnv, seedCells, seeded, seedView, seedStates, seedUntouched, seedValues⟩ :=
    SeedStates.result (globals := Domain.bindings primeAddress) mapHeap mapped
  let seededHeap := ProjectEulerSolutions.HeapCollection.working mapHeap
    (ProjectEulerSolutions.MaskedStates.references seedCells) seedGarbage
  have seededMaps : BuildMaps.MapsView seededHeap heap.length entries :=
    mapsView.append (.list (ProjectEulerSolutions.MaskedStates.references seedCells) :: seedGarbage)
  have seedLong : ∀ state ∈ seeds, 2 ≤ state.1.length := by
    intro state member
    obtain ⟨x, _, rfl⟩ := List.mem_map.mp member
    simp [digits]
  have storedLong : ∀ cell ∈ seedCells, 2 ≤ cell.state.1.length := by
    intro cell member
    apply seedLong cell.state
    rw [← seedValues]
    exact List.mem_map_of_mem (f := ProjectEulerSolutions.MaskedStates.Cell.state) member
  have present : ∀ p ∈ ExtendStates.primes, p ∈ entries.map BuildMaps.PrimeTable.modulus := by
    intro p member
    rw [coverage]
    simp only [ExtendStates.primes, List.mem_cons, List.not_mem_nil, or_false] at member
    simp only [primes, List.mem_cons, List.not_mem_nil, or_false]
    tauto
  obtain ⟨extensionExtra, extendedEnv, cells, statesAddress, extended, finalStates, statesFound, stored, _, values⟩ :=
    ExtendStates.result (globals := Domain.bindings primeAddress) seededHeap heap.length entries seededMaps present
      seedCells mapHeap.length (ProjectEulerSolutions.HeapCollection.found _ _ _) seedView.stored storedLong seededEnv
      (by rw [seedUntouched "maps" (by decide)]; simp [mapped, setLocal]) seedStates
  have stateValues : cells.map ProjectEulerSolutions.MaskedStates.Cell.state = states := by
    simpa only [seedValues, ExtendStates.primes, states] using values
  obtain ⟨finalEnv, completed, total⟩ := CompleteStates.result (globals := Domain.bindings primeAddress)
    cells (seededHeap ++ extensionExtra) statesAddress statesFound stored stateValues extendedEnv finalStates
  let extra := mapsExtra ++ (.list (ProjectEulerSolutions.MaskedStates.references seedCells) :: seedGarbage) ++ extensionExtra
  have finalHeap : seededHeap ++ extensionExtra = heap ++ extra := by
    simp [seededHeap, mapHeap, extra, ProjectEulerSolutions.HeapCollection.working, List.append_assoc]
  refine ⟨extra, ?_⟩
  rw [← finalHeap]
  apply BlockReturns.function rfl rfl
  apply (((setMaps.append seeded).append extended).append completed).returns
  exact BlockReturns.return_value (Evaluates.lookup total)

end solve

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (primeAddress : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref primeAddress) "list[int]")
    (requires : HeapCalls.Condition program (Domain.env primeAddress) Solvers.P43Ast.requires_solve_0 heap
      (Domain.bindings primeAddress)) :
    (SetCalls.call program "solve" [] heap (Domain.bindings primeAddress)).Equiv
      (SetCalls.call P43Ast.module "naive" [] heap (Domain.bindings primeAddress)) := by
  have found := Domain.initialized_of_condition heap primeAddress typed requires
  obtain ⟨extra, optimized⟩ := solve.result heap primeAddress found
  obtain ⟨referenceExtra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p43.naive.result
      (globals := Domain.bindings primeAddress) heap
  rw [ProjectEulerSolutions.SubstringSearch.sum_equiv] at optimized
  intro leftChoices rightChoices
  apply Mutable.Call.equiv_of_finishes ((SetCalls.ReturnsValue.of_heap optimized).finishes leftChoices)
    ((SetCalls.ReturnsValue.of_heap reference).finishes rightChoices)
  apply Mutable.Completion.equivalent_allocations heap extra referenceExtra
    (.scalar (.int (ProjectEulerSolutions.SubstringDivisibility.referenceRows.map ProjectEulerSolutions.SubstringDivisibility.value).sum))
    (Domain.bindings primeAddress) valid trivial
  have within := (List.getElem?_eq_some_iff.mp found).1
  simpa [Domain.bindings, Mutable.Value.edge, Graph.Edge.Within] using within

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (primeAddress : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref primeAddress) "list[int]" →
      HeapCalls.Condition program (Domain.env primeAddress) Solvers.P43Ast.requires_solve_0 heap (Domain.bindings primeAddress) →
      (SetCalls.call program "solve" [] heap (Domain.bindings primeAddress)).Supported) ∧
    (∀ (heap : Mutable.Heap) (primeAddress : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref primeAddress) "list[int]" →
      HeapCalls.Condition program (Domain.env primeAddress) Solvers.P43Ast.requires_solve_0 heap (Domain.bindings primeAddress) →
      (SetCalls.call P43Ast.module "naive" [] heap (Domain.bindings primeAddress)).Supported) := by
  constructor
  · intro heap primeAddress _ typed requires
    have found := Domain.initialized_of_condition heap primeAddress typed requires
    obtain ⟨_, optimized⟩ := solve.result heap primeAddress found
    intro choices
    exact Execution.supported_of_finishes ((SetCalls.ReturnsValue.of_heap optimized).finishes choices)
  · intro heap primeAddress _ _ _ choices
    obtain ⟨_, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p43.naive.result
        (globals := Domain.bindings primeAddress) heap
    exact Execution.supported_of_finishes ((SetCalls.ReturnsValue.of_heap reference).finishes choices)

end solvers.«43»
