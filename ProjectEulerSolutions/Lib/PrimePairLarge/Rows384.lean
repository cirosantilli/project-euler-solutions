import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12919_small : List Nat := []
theorem row12919_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12919 row12919_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12919_block_005 : List Nat := []
theorem row12919_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12919 row12919_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12919_block_006 : List Nat := []
theorem row12919_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12919 row12919_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12919_block_007 : List Nat := []
def plan12919_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12919_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12919 row12919_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12919_block_007
  · decide
  · rfl

def row12919_block_008 : List Nat := []
theorem row12919_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12919 row12919_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12919_block_009 : List Nat := []
theorem row12919_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12919 row12919_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12919_block_010 : List Nat := []
theorem row12919_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12919 row12919_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12919_block_011 : List Nat := []
theorem row12919_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12919 row12919_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12919_block_012 : List Nat := []
theorem row12919_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12919 row12919_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12919_block_013 : List Nat := []
theorem row12919_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12919 row12919_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12919_large_left_left : List Nat := row12919_small ++ row12919_block_005
theorem row12919_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12919 row12919_large_left_left :=
  Budget.RowUpperBound.branch row12919_small_covers row12919_block_005_covers

def row12919_large_left_right_right : List Nat := row12919_block_007 ++ row12919_block_008
theorem row12919_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12919 row12919_large_left_right_right :=
  Budget.RowUpperBound.branch row12919_block_007_covers row12919_block_008_covers

def row12919_large_left_right : List Nat := row12919_block_006 ++ row12919_large_left_right_right
theorem row12919_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12919 row12919_large_left_right :=
  Budget.RowUpperBound.branch row12919_block_006_covers row12919_large_left_right_right_covers

def row12919_large_left : List Nat := row12919_large_left_left ++ row12919_large_left_right
theorem row12919_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12919 row12919_large_left :=
  Budget.RowUpperBound.branch row12919_large_left_left_covers row12919_large_left_right_covers

def row12919_large_right_left : List Nat := row12919_block_009 ++ row12919_block_010
theorem row12919_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12919 row12919_large_right_left :=
  Budget.RowUpperBound.branch row12919_block_009_covers row12919_block_010_covers

def row12919_large_right_right_right : List Nat := row12919_block_012 ++ row12919_block_013
theorem row12919_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12919 row12919_large_right_right_right :=
  Budget.RowUpperBound.branch row12919_block_012_covers row12919_block_013_covers

def row12919_large_right_right : List Nat := row12919_block_011 ++ row12919_large_right_right_right
theorem row12919_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12919 row12919_large_right_right :=
  Budget.RowUpperBound.branch row12919_block_011_covers row12919_large_right_right_right_covers

def row12919_large_right : List Nat := row12919_large_right_left ++ row12919_large_right_right
theorem row12919_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12919 row12919_large_right :=
  Budget.RowUpperBound.branch row12919_large_right_left_covers row12919_large_right_right_covers

def row12919_large : List Nat := row12919_large_left ++ row12919_large_right
theorem row12919_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12919 row12919_large :=
  Budget.RowUpperBound.branch row12919_large_left_covers row12919_large_right_covers

def row12919 : List Nat := row12919_large
theorem row12919_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12919 row12919 := row12919_large_covers

def row12923_small : List Nat := []
theorem row12923_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12923 row12923_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12923_block_005 : List Nat := []
theorem row12923_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12923 row12923_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12923_block_006 : List Nat := []
theorem row12923_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12923 row12923_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12923_block_007 : List Nat := []
def plan12923_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12923_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12923 row12923_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12923_block_007
  · decide
  · rfl

def row12923_block_008 : List Nat := []
theorem row12923_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12923 row12923_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12923_block_009 : List Nat := []
theorem row12923_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12923 row12923_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12923_block_010 : List Nat := []
theorem row12923_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12923 row12923_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12923_block_011 : List Nat := []
theorem row12923_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12923 row12923_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12923_block_012 : List Nat := []
theorem row12923_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12923 row12923_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12923_block_013 : List Nat := []
theorem row12923_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12923 row12923_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12923_large_left_left : List Nat := row12923_small ++ row12923_block_005
theorem row12923_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12923 row12923_large_left_left :=
  Budget.RowUpperBound.branch row12923_small_covers row12923_block_005_covers

def row12923_large_left_right_right : List Nat := row12923_block_007 ++ row12923_block_008
theorem row12923_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12923 row12923_large_left_right_right :=
  Budget.RowUpperBound.branch row12923_block_007_covers row12923_block_008_covers

def row12923_large_left_right : List Nat := row12923_block_006 ++ row12923_large_left_right_right
theorem row12923_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12923 row12923_large_left_right :=
  Budget.RowUpperBound.branch row12923_block_006_covers row12923_large_left_right_right_covers

def row12923_large_left : List Nat := row12923_large_left_left ++ row12923_large_left_right
theorem row12923_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12923 row12923_large_left :=
  Budget.RowUpperBound.branch row12923_large_left_left_covers row12923_large_left_right_covers

def row12923_large_right_left : List Nat := row12923_block_009 ++ row12923_block_010
theorem row12923_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12923 row12923_large_right_left :=
  Budget.RowUpperBound.branch row12923_block_009_covers row12923_block_010_covers

def row12923_large_right_right_right : List Nat := row12923_block_012 ++ row12923_block_013
theorem row12923_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12923 row12923_large_right_right_right :=
  Budget.RowUpperBound.branch row12923_block_012_covers row12923_block_013_covers

def row12923_large_right_right : List Nat := row12923_block_011 ++ row12923_large_right_right_right
theorem row12923_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12923 row12923_large_right_right :=
  Budget.RowUpperBound.branch row12923_block_011_covers row12923_large_right_right_right_covers

def row12923_large_right : List Nat := row12923_large_right_left ++ row12923_large_right_right
theorem row12923_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12923 row12923_large_right :=
  Budget.RowUpperBound.branch row12923_large_right_left_covers row12923_large_right_right_covers

def row12923_large : List Nat := row12923_large_left ++ row12923_large_right
theorem row12923_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12923 row12923_large :=
  Budget.RowUpperBound.branch row12923_large_left_covers row12923_large_right_covers

def row12923 : List Nat := row12923_large
theorem row12923_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12923 row12923 := row12923_large_covers

def row12941_small : List Nat := []
theorem row12941_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12941 row12941_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12941_block_005 : List Nat := []
theorem row12941_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12941 row12941_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12941_block_006 : List Nat := []
theorem row12941_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12941 row12941_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12941_block_007 : List Nat := []
def plan12941_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12941_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12941 row12941_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12941_block_007
  · decide
  · rfl

def row12941_block_008 : List Nat := []
theorem row12941_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12941 row12941_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12941_block_009 : List Nat := []
theorem row12941_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12941 row12941_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12941_block_010 : List Nat := []
theorem row12941_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12941 row12941_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12941_block_011 : List Nat := []
theorem row12941_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12941 row12941_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12941_block_012 : List Nat := []
theorem row12941_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12941 row12941_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12941_block_013 : List Nat := []
theorem row12941_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12941 row12941_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12941_large_left_left : List Nat := row12941_small ++ row12941_block_005
theorem row12941_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12941 row12941_large_left_left :=
  Budget.RowUpperBound.branch row12941_small_covers row12941_block_005_covers

def row12941_large_left_right_right : List Nat := row12941_block_007 ++ row12941_block_008
theorem row12941_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12941 row12941_large_left_right_right :=
  Budget.RowUpperBound.branch row12941_block_007_covers row12941_block_008_covers

def row12941_large_left_right : List Nat := row12941_block_006 ++ row12941_large_left_right_right
theorem row12941_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12941 row12941_large_left_right :=
  Budget.RowUpperBound.branch row12941_block_006_covers row12941_large_left_right_right_covers

def row12941_large_left : List Nat := row12941_large_left_left ++ row12941_large_left_right
theorem row12941_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12941 row12941_large_left :=
  Budget.RowUpperBound.branch row12941_large_left_left_covers row12941_large_left_right_covers

def row12941_large_right_left : List Nat := row12941_block_009 ++ row12941_block_010
theorem row12941_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12941 row12941_large_right_left :=
  Budget.RowUpperBound.branch row12941_block_009_covers row12941_block_010_covers

def row12941_large_right_right_right : List Nat := row12941_block_012 ++ row12941_block_013
theorem row12941_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12941 row12941_large_right_right_right :=
  Budget.RowUpperBound.branch row12941_block_012_covers row12941_block_013_covers

def row12941_large_right_right : List Nat := row12941_block_011 ++ row12941_large_right_right_right
theorem row12941_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12941 row12941_large_right_right :=
  Budget.RowUpperBound.branch row12941_block_011_covers row12941_large_right_right_right_covers

def row12941_large_right : List Nat := row12941_large_right_left ++ row12941_large_right_right
theorem row12941_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12941 row12941_large_right :=
  Budget.RowUpperBound.branch row12941_large_right_left_covers row12941_large_right_right_covers

def row12941_large : List Nat := row12941_large_left ++ row12941_large_right
theorem row12941_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12941 row12941_large :=
  Budget.RowUpperBound.branch row12941_large_left_covers row12941_large_right_covers

def row12941 : List Nat := row12941_large
theorem row12941_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12941 row12941 := row12941_large_covers

def row12953_small : List Nat := []
theorem row12953_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12953 row12953_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12953_block_005 : List Nat := []
theorem row12953_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12953 row12953_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12953_block_006 : List Nat := []
theorem row12953_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12953 row12953_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12953_block_007 : List Nat := [13001]
def plan12953_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 31), .first, .first, .first, (.firstFactor 37), .possible, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12953_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12953 row12953_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12953_block_007
  · decide
  · rfl

def row12953_block_008 : List Nat := []
theorem row12953_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12953 row12953_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12953_block_009 : List Nat := []
theorem row12953_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12953 row12953_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12953_block_010 : List Nat := []
theorem row12953_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12953 row12953_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12953_block_011 : List Nat := []
theorem row12953_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12953 row12953_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12953_block_012 : List Nat := []
theorem row12953_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12953 row12953_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12953_block_013 : List Nat := []
theorem row12953_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12953 row12953_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12953_large_left_left : List Nat := row12953_small ++ row12953_block_005
theorem row12953_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12953 row12953_large_left_left :=
  Budget.RowUpperBound.branch row12953_small_covers row12953_block_005_covers

def row12953_large_left_right_right : List Nat := row12953_block_007 ++ row12953_block_008
theorem row12953_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12953 row12953_large_left_right_right :=
  Budget.RowUpperBound.branch row12953_block_007_covers row12953_block_008_covers

def row12953_large_left_right : List Nat := row12953_block_006 ++ row12953_large_left_right_right
theorem row12953_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12953 row12953_large_left_right :=
  Budget.RowUpperBound.branch row12953_block_006_covers row12953_large_left_right_right_covers

def row12953_large_left : List Nat := row12953_large_left_left ++ row12953_large_left_right
theorem row12953_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12953 row12953_large_left :=
  Budget.RowUpperBound.branch row12953_large_left_left_covers row12953_large_left_right_covers

def row12953_large_right_left : List Nat := row12953_block_009 ++ row12953_block_010
theorem row12953_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12953 row12953_large_right_left :=
  Budget.RowUpperBound.branch row12953_block_009_covers row12953_block_010_covers

def row12953_large_right_right_right : List Nat := row12953_block_012 ++ row12953_block_013
theorem row12953_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12953 row12953_large_right_right_right :=
  Budget.RowUpperBound.branch row12953_block_012_covers row12953_block_013_covers

def row12953_large_right_right : List Nat := row12953_block_011 ++ row12953_large_right_right_right
theorem row12953_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12953 row12953_large_right_right :=
  Budget.RowUpperBound.branch row12953_block_011_covers row12953_large_right_right_right_covers

def row12953_large_right : List Nat := row12953_large_right_left ++ row12953_large_right_right
theorem row12953_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12953 row12953_large_right :=
  Budget.RowUpperBound.branch row12953_large_right_left_covers row12953_large_right_right_covers

def row12953_large : List Nat := row12953_large_left ++ row12953_large_right
theorem row12953_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12953 row12953_large :=
  Budget.RowUpperBound.branch row12953_large_left_covers row12953_large_right_covers

def row12953 : List Nat := row12953_large
theorem row12953_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12953 row12953 := row12953_large_covers

end ProjectEulerSolutions.PrimePairLarge
