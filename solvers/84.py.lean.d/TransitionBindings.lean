import solvers.«84.py.lean.d».TransitionLanding
import solvers.«84.py.lean.d».LandingWeights
import lang.py.FormalSW.UnpackingCalls

namespace solvers.«84».transition_row.bindings
open FormalSW.PyAst HeapCalls resolve_landing
set_option Elab.async false

def target : Expr := .tuple [.tuple [.name "destination", .name "sent"], .name "weight"]
def body : List Stmt := [.assign [target] (.name "$formalsw.unpack")]

def locals (before : HeapCalls.Env) (entry : weights.Entry) : HeapCalls.Env :=
  let outer := assignNames before ["$formalsw.target.0", "$formalsw.target.1"]
    [.ref entry.address, .scalar (.int entry.weight)]
  let inner := assignNames outer ["$formalsw.target.0.0", "$formalsw.target.0.1"]
    [.scalar (.int entry.landing.1), .scalar (.bool entry.landing.2)]
  setLocal (setLocal (setLocal inner "destination" (.scalar (.int entry.landing.1))) "sent" (.scalar (.bool entry.landing.2)))
    "weight" (.scalar (.int entry.weight))

/-- The nested target binds the original cached key's two fields without copying it. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (pairAddress : Nat) (entry : weights.Entry)
    (lookup : before "$formalsw.unpack" = some (.ref pairAddress))
    (pair : heap[pairAddress]? = some (.tuple [.ref entry.address, .scalar (.int entry.weight)]))
    (key : heap[entry.address]? = some (.tuple [.scalar (.int entry.landing.1), .scalar (.bool entry.landing.2)])) :
    BlockContinues program before body heap (locals before entry) heap globals := by
  apply BlockContinues.unpack_nested false rfl
  have outer := BlockContinues.unpack (module := program) (globals := globals)
    (heap := heap) (middle := heap) (finalHeap := heap)
    (items := [.ref entry.address, .scalar (.int entry.weight)])
    (expressions := [.name "$formalsw.target.0", .name "$formalsw.target.1"])
    (names := ["$formalsw.target.0", "$formalsw.target.1"]) rfl
    (Evaluates.lookup lookup) (by simp [Mutable.sequence, pair]) rfl
  have inner := BlockContinues.unpack (module := program) (globals := globals)
    (heap := heap) (middle := heap) (finalHeap := heap)
    (items := [.scalar (.int entry.landing.1), .scalar (.bool entry.landing.2)]) (value := .ref entry.address)
    (expressions := [.name "$formalsw.target.0.0", .name "$formalsw.target.0.1"])
    (names := ["$formalsw.target.0.0", "$formalsw.target.0.1"]) rfl
    (Evaluates.lookup (name := "$formalsw.target.0") (heap := heap)
      (env := assignNames before ["$formalsw.target.0", "$formalsw.target.1"]
        [.ref entry.address, .scalar (.int entry.weight)]) (by simp [assignNames, setLocal]))
    (by simp [Mutable.sequence, key]) rfl
  simpa [UnpackingTargets.assignment, UnpackingTargets.lower, UnpackingTargets.children,
    UnpackingTargets.names, UnpackingTargets.temporary, List.range_succ, locals] using
    outer.append (inner.append
    ((BlockContinues.assign (name := "destination")
      (Evaluates.lookup (name := "$formalsw.target.0.0") rfl)).append
    ((BlockContinues.assign (name := "sent")
      (Evaluates.lookup (name := "$formalsw.target.0.1") rfl)).append
    ((BlockContinues.assign (name := "weight")
      (Evaluates.lookup (name := "$formalsw.target.1") rfl)).append BlockContinues.nil))))

def rebound : List String :=
  ["$formalsw.target.0", "$formalsw.target.1", "$formalsw.target.0.0", "$formalsw.target.0.1",
    "destination", "sent", "weight"]

theorem unchanged (before : HeapCalls.Env) (entry : weights.Entry) (name : String) (keep : name ∉ rebound) :
    locals before entry name = before name := by
  simp only [rebound, List.mem_cons, not_or] at keep
  simp_all [locals, assignNames, setLocal]

end solvers.«84».transition_row.bindings
