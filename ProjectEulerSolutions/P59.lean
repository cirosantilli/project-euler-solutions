import ProjectEulerStatements.P59
import ProjectEulerSolutions.Lib.XorDecryption

namespace ProjectEulerSolutions.P59

def parseCipher (text : String) : List Nat :=
  (text.trimAscii.toString.splitOn ",").filterMap String.toNat?

/-- Decrypt, reject nonprintable text, and accumulate score and sum in one pass per key. -/
def solve (cipher : List Nat) : Nat :=
  (XorDecryption.selectedSum (XorDecryption.scan
    (XorDecryption.scoreKeyStreaming (cipher.map Nat.cast)) XorDecryption.keys none)).toNat

end ProjectEulerSolutions.P59
