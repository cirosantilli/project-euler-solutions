import ProjectEulerSolutions.Lib.TotientSieveHeap
import lang.py.FormalSW.LoopCalls

/-! The inner source loop implements the bounded mathematical prime-product pass. -/
namespace ProjectEulerSolutions.TotientSieveCalls
open FormalSW.PyAst HeapCalls ProjectEulerStatements.TotientLinearSieve
set_option Elab.async false
variable {program : Module}

abbrev productExpression : Expr := .binOp (.name "i") .mult (.name "p")
abbrev boundCondition : Expr := .compare (.name "ip") [(.gt, .name "limit")]
abbrev dividesCondition : Expr :=
  .compare (.binOp (.name "i") .mod (.name "p")) [(.eq, .constant (.int 0))]
abbrev repeatedExpression : Expr :=
  .binOp (.subscript (.name "phi") (.name "i")) .mult (.name "p")
abbrev freshExpression : Expr := .binOp (.subscript (.name "phi") (.name "i")) .mult
  (.binOp (.name "p") .sub (.constant (.int 1)))
abbrev markCode : Stmt :=
  .assign [.subscript (.name "is_comp") (.name "ip")] (.constant (.int 1))
abbrev repeatedCode : Stmt :=
  .assign [.subscript (.name "phi") (.name "ip")] repeatedExpression
abbrev freshCode : Stmt :=
  .assign [.subscript (.name "phi") (.name "ip")] freshExpression
abbrev innerBody : List Stmt :=
  [.assign [.name "ip"] productExpression, .ifThen boundCondition [.breakLoop] [],
    markCode, .ifThen dividesCondition [repeatedCode, .breakLoop] [], freshCode]
abbrev innerLoop : Stmt := .forIn (.name "p") (.name "primes") innerBody []

theorem inner_source : innerBody =
    [.assign [.name "ip"] productExpression, .ifThen boundCondition [.breakLoop] [],
      markCode, .ifThen dividesCondition [repeatedCode, .breakLoop] [], freshCode] := rfl

structure Ready (limit i address : Nat) (env : HeapCalls.Env) : Prop where
  getLimit : env "limit" = some (number limit)
  getI : env "i" = some (number i)
  getPhi : env "phi" = some (.ref (address + 1))
  getMarks : env "is_comp" = some (.ref (address + 3))
  getCache : env "primes" = some (.ref (address + 4))

theorem Ready.set_p {limit i address : Nat} {env : HeapCalls.Env}
    (ready : Ready limit i address env) (p : Nat) :
    Ready limit i address (setLocal env "p" (number p)) where
  getLimit := by simpa [setLocal] using ready.getLimit
  getI := by simpa [setLocal] using ready.getI
  getPhi := by simpa [setLocal] using ready.getPhi
  getMarks := by simpa [setLocal] using ready.getMarks
  getCache := by simpa [setLocal] using ready.getCache

theorem Ready.set_ip {limit i address : Nat} {env : HeapCalls.Env}
    (ready : Ready limit i address env) (ip : Nat) :
    Ready limit i address (setLocal env "ip" (number ip)) where
  getLimit := by simpa [setLocal] using ready.getLimit
  getI := by simpa [setLocal] using ready.getI
  getPhi := by simpa [setLocal] using ready.getPhi
  getMarks := by simpa [setLocal] using ready.getMarks
  getCache := by simpa [setLocal] using ready.getCache

theorem product_result (i p : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getI : env "i" = some (number i)) (getP : env "p" = some (number p)) :
    Evaluates program env productExpression heap (number (i * p)) heap :=
  Evaluates.binary (Evaluates.lookup getI) (Evaluates.lookup getP)
    (by simp [Mutable.binary, scalarBinary])

theorem bound_result (limit ip : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) (getIP : env "ip" = some (number ip)) :
    Evaluates program env boundCondition heap (.scalar (.bool (decide (limit < ip)))) heap := by
  apply Evaluates.compare (Evaluates.lookup getIP) (Evaluates.lookup getLimit)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem divides_result (i p : Nat) (prime : Nat.Prime p) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (getI : env "i" = some (number i))
    (getP : env "p" = some (number p)) :
    Evaluates program env dividesCondition heap (.scalar (.bool (decide (i % p = 0)))) heap := by
  apply Evaluates.compare (av := number (i % p)) (bv := number 0)
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (middle := heap) (afterRight := heap)
      (Evaluates.lookup getI) (Evaluates.lookup getP)
      (by simp [Mutable.binary, scalarBinary, prime.pos, prime.ne_zero])
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    norm_cast

theorem repeated_result (limit i p : Nat) (table : Table) (base : Mutable.Heap)
    (marked : Nat → Bool) (cache : List Nat) (env : HeapCalls.Env)
    (within : i ≤ limit) (ready : Ready limit i base.length env)
    (getP : env "p" = some (number p)) :
    Evaluates program env repeatedExpression
      (working base (limit + 1) (fun n => (table n).1) marked cache)
      (number ((table i).1 * p))
      (working base (limit + 1) (fun n => (table n).1) marked cache) := by
  exact Evaluates.binary
    (read_phi (program := program)
      base (limit + 1) i _ marked cache env "i" (by omega) ready.getPhi ready.getI)
    (Evaluates.lookup getP) (by simp [Mutable.binary, scalarBinary])

theorem fresh_result (limit i p : Nat) (prime : Nat.Prime p) (table : Table)
    (base : Mutable.Heap) (marked : Nat → Bool) (cache : List Nat) (env : HeapCalls.Env)
    (within : i ≤ limit) (ready : Ready limit i base.length env)
    (getP : env "p" = some (number p)) :
    Evaluates program env freshExpression
      (working base (limit + 1) (fun n => (table n).1) marked cache)
      (number ((table i).1 * (p - 1)))
      (working base (limit + 1) (fun n => (table n).1) marked cache) := by
  apply Evaluates.binary (av := number ((table i).1)) (bv := number (p - 1))
    (middle := working base (limit + 1) (fun n => (table n).1) marked cache)
    (afterRight := working base (limit + 1) (fun n => (table n).1) marked cache)
    (read_phi (program := program)
      base (limit + 1) i _ marked cache env "i" (by omega) ready.getPhi ready.getI)
  · exact Evaluates.binary (Evaluates.lookup getP) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number, Nat.cast_sub prime.one_lt.le])
  · simp [Mutable.binary, scalarBinary]

def writtenValue (table : Table) (i p : Nat) : Nat :=
  if i % p = 0 then (table i).1 * p else (table i).1 * (p - 1)

theorem write_heap (base : Mutable.Heap) (size i p : Nat) (table : Table) (cache : List Nat) :
    working base size (Function.update (fun n => (table n).1) (i * p) (writtenValue table i p))
      (Function.update (fun n => (table n).2) (i * p) true) cache =
        realize base size (write i p table) cache := by
  have first : Function.update (fun n => (table n).1) (i * p) (writtenValue table i p) =
      fun n => (write i p table n).1 := by
    funext n
    by_cases same : n = i * p
    · subst n
      simp only [Function.update_self, write]
      by_cases divides : i % p = 0 <;> simp [writtenValue, divides]
    · simp [write, Function.update_of_ne same]
  have second : Function.update (fun n => (table n).2) (i * p) true =
      fun n => (write i p table n).2 := by
    funext n
    by_cases same : n = i * p
    · subst n
      simp only [Function.update_self, write]
      split <;> rfl
    · simp [write, Function.update_of_ne same]
  rw [first, second]

end ProjectEulerSolutions.TotientSieveCalls
