import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12113_small : List Nat := []
theorem row12113_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12113 row12113_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12113_block_005 : List Nat := []
theorem row12113_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12113 row12113_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12113_block_006 : List Nat := [12251]
def plan12113_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 23), (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 23), (.secondFactor 19), .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.secondFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row12113_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12113 row12113_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12113_block_006
  · decide
  · rfl

def row12113_block_007 : List Nat := [13457, 13679]
def plan12113_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 13), (.firstFactor 23), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .possible, (.secondFactor 31), (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12113_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12113 row12113_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12113_block_007
  · decide
  · rfl

def row12113_block_008 : List Nat := []
theorem row12113_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12113 row12113_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12113_block_009 : List Nat := []
theorem row12113_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12113 row12113_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12113_block_010 : List Nat := []
theorem row12113_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12113 row12113_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12113_block_011 : List Nat := []
theorem row12113_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12113 row12113_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12113_block_012 : List Nat := []
theorem row12113_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12113 row12113_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12113_block_013 : List Nat := []
theorem row12113_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12113 row12113_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12113_large_left_left : List Nat := row12113_small ++ row12113_block_005
theorem row12113_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12113 row12113_large_left_left :=
  Budget.RowUpperBound.branch row12113_small_covers row12113_block_005_covers

def row12113_large_left_right_right : List Nat := row12113_block_007 ++ row12113_block_008
theorem row12113_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12113 row12113_large_left_right_right :=
  Budget.RowUpperBound.branch row12113_block_007_covers row12113_block_008_covers

def row12113_large_left_right : List Nat := row12113_block_006 ++ row12113_large_left_right_right
theorem row12113_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12113 row12113_large_left_right :=
  Budget.RowUpperBound.branch row12113_block_006_covers row12113_large_left_right_right_covers

def row12113_large_left : List Nat := row12113_large_left_left ++ row12113_large_left_right
theorem row12113_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12113 row12113_large_left :=
  Budget.RowUpperBound.branch row12113_large_left_left_covers row12113_large_left_right_covers

def row12113_large_right_left : List Nat := row12113_block_009 ++ row12113_block_010
theorem row12113_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12113 row12113_large_right_left :=
  Budget.RowUpperBound.branch row12113_block_009_covers row12113_block_010_covers

def row12113_large_right_right_right : List Nat := row12113_block_012 ++ row12113_block_013
theorem row12113_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12113 row12113_large_right_right_right :=
  Budget.RowUpperBound.branch row12113_block_012_covers row12113_block_013_covers

def row12113_large_right_right : List Nat := row12113_block_011 ++ row12113_large_right_right_right
theorem row12113_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12113 row12113_large_right_right :=
  Budget.RowUpperBound.branch row12113_block_011_covers row12113_large_right_right_right_covers

def row12113_large_right : List Nat := row12113_large_right_left ++ row12113_large_right_right
theorem row12113_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12113 row12113_large_right :=
  Budget.RowUpperBound.branch row12113_large_right_left_covers row12113_large_right_right_covers

def row12113_large : List Nat := row12113_large_left ++ row12113_large_right
theorem row12113_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12113 row12113_large :=
  Budget.RowUpperBound.branch row12113_large_left_covers row12113_large_right_covers

def row12113 : List Nat := row12113_large
theorem row12113_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12113 row12113 := row12113_large_covers

def row12119_small : List Nat := []
theorem row12119_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12119 row12119_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12119_block_005 : List Nat := []
theorem row12119_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12119 row12119_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12119_block_006 : List Nat := []
def plan12119_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second]
set_option maxRecDepth 2048 in
theorem row12119_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12119 row12119_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12119_block_006
  · decide
  · rfl

def row12119_block_007 : List Nat := [12743, 12917, 13331, 13463]
def plan12119_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 23), .second, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .second, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 19), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .second, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12119_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12119 row12119_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12119_block_007
  · decide
  · rfl

def row12119_block_008 : List Nat := []
theorem row12119_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12119 row12119_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12119_block_009 : List Nat := []
theorem row12119_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12119 row12119_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12119_block_010 : List Nat := []
theorem row12119_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12119 row12119_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12119_block_011 : List Nat := []
theorem row12119_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12119 row12119_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12119_block_012 : List Nat := []
theorem row12119_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12119 row12119_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12119_block_013 : List Nat := []
theorem row12119_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12119 row12119_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12119_large_left_left : List Nat := row12119_small ++ row12119_block_005
theorem row12119_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12119 row12119_large_left_left :=
  Budget.RowUpperBound.branch row12119_small_covers row12119_block_005_covers

def row12119_large_left_right_right : List Nat := row12119_block_007 ++ row12119_block_008
theorem row12119_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12119 row12119_large_left_right_right :=
  Budget.RowUpperBound.branch row12119_block_007_covers row12119_block_008_covers

def row12119_large_left_right : List Nat := row12119_block_006 ++ row12119_large_left_right_right
theorem row12119_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12119 row12119_large_left_right :=
  Budget.RowUpperBound.branch row12119_block_006_covers row12119_large_left_right_right_covers

def row12119_large_left : List Nat := row12119_large_left_left ++ row12119_large_left_right
theorem row12119_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12119 row12119_large_left :=
  Budget.RowUpperBound.branch row12119_large_left_left_covers row12119_large_left_right_covers

def row12119_large_right_left : List Nat := row12119_block_009 ++ row12119_block_010
theorem row12119_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12119 row12119_large_right_left :=
  Budget.RowUpperBound.branch row12119_block_009_covers row12119_block_010_covers

def row12119_large_right_right_right : List Nat := row12119_block_012 ++ row12119_block_013
theorem row12119_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12119 row12119_large_right_right_right :=
  Budget.RowUpperBound.branch row12119_block_012_covers row12119_block_013_covers

def row12119_large_right_right : List Nat := row12119_block_011 ++ row12119_large_right_right_right
theorem row12119_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12119 row12119_large_right_right :=
  Budget.RowUpperBound.branch row12119_block_011_covers row12119_large_right_right_right_covers

def row12119_large_right : List Nat := row12119_large_right_left ++ row12119_large_right_right
theorem row12119_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12119 row12119_large_right :=
  Budget.RowUpperBound.branch row12119_large_right_left_covers row12119_large_right_right_covers

def row12119_large : List Nat := row12119_large_left ++ row12119_large_right
theorem row12119_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12119 row12119_large :=
  Budget.RowUpperBound.branch row12119_large_left_covers row12119_large_right_covers

def row12119 : List Nat := row12119_large
theorem row12119_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12119 row12119 := row12119_large_covers

def row12143_small : List Nat := []
theorem row12143_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12143 row12143_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12143_block_005 : List Nat := []
theorem row12143_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12143 row12143_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12143_block_006 : List Nat := [12269]
def plan12143_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 37), .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row12143_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12143 row12143_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12143_block_006
  · decide
  · rfl

def row12143_block_007 : List Nat := [12791]
def plan12143_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 29), .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12143_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12143 row12143_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12143_block_007
  · decide
  · rfl

def row12143_block_008 : List Nat := []
theorem row12143_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12143 row12143_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12143_block_009 : List Nat := []
theorem row12143_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12143 row12143_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12143_block_010 : List Nat := []
theorem row12143_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12143 row12143_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12143_block_011 : List Nat := []
theorem row12143_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12143 row12143_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12143_block_012 : List Nat := []
theorem row12143_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12143 row12143_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12143_block_013 : List Nat := []
theorem row12143_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12143 row12143_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12143_large_left_left : List Nat := row12143_small ++ row12143_block_005
theorem row12143_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12143 row12143_large_left_left :=
  Budget.RowUpperBound.branch row12143_small_covers row12143_block_005_covers

def row12143_large_left_right_right : List Nat := row12143_block_007 ++ row12143_block_008
theorem row12143_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12143 row12143_large_left_right_right :=
  Budget.RowUpperBound.branch row12143_block_007_covers row12143_block_008_covers

def row12143_large_left_right : List Nat := row12143_block_006 ++ row12143_large_left_right_right
theorem row12143_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12143 row12143_large_left_right :=
  Budget.RowUpperBound.branch row12143_block_006_covers row12143_large_left_right_right_covers

def row12143_large_left : List Nat := row12143_large_left_left ++ row12143_large_left_right
theorem row12143_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12143 row12143_large_left :=
  Budget.RowUpperBound.branch row12143_large_left_left_covers row12143_large_left_right_covers

def row12143_large_right_left : List Nat := row12143_block_009 ++ row12143_block_010
theorem row12143_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12143 row12143_large_right_left :=
  Budget.RowUpperBound.branch row12143_block_009_covers row12143_block_010_covers

def row12143_large_right_right_right : List Nat := row12143_block_012 ++ row12143_block_013
theorem row12143_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12143 row12143_large_right_right_right :=
  Budget.RowUpperBound.branch row12143_block_012_covers row12143_block_013_covers

def row12143_large_right_right : List Nat := row12143_block_011 ++ row12143_large_right_right_right
theorem row12143_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12143 row12143_large_right_right :=
  Budget.RowUpperBound.branch row12143_block_011_covers row12143_large_right_right_right_covers

def row12143_large_right : List Nat := row12143_large_right_left ++ row12143_large_right_right
theorem row12143_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12143 row12143_large_right :=
  Budget.RowUpperBound.branch row12143_large_right_left_covers row12143_large_right_right_covers

def row12143_large : List Nat := row12143_large_left ++ row12143_large_right
theorem row12143_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12143 row12143_large :=
  Budget.RowUpperBound.branch row12143_large_left_covers row12143_large_right_covers

def row12143 : List Nat := row12143_large
theorem row12143_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12143 row12143 := row12143_large_covers

def row12149_small : List Nat := []
theorem row12149_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12149 row12149_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12149_block_005 : List Nat := []
theorem row12149_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12149 row12149_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12149_block_006 : List Nat := []
def plan12149_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row12149_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12149 row12149_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12149_block_006
  · decide
  · rfl

def row12149_block_007 : List Nat := [12659, 12821, 13829]
def plan12149_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .second, .possible, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12149_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12149 row12149_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12149_block_007
  · decide
  · rfl

def row12149_block_008 : List Nat := []
theorem row12149_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12149 row12149_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12149_block_009 : List Nat := []
theorem row12149_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12149 row12149_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12149_block_010 : List Nat := []
theorem row12149_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12149 row12149_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12149_block_011 : List Nat := []
theorem row12149_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12149 row12149_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12149_block_012 : List Nat := []
theorem row12149_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12149 row12149_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12149_block_013 : List Nat := []
theorem row12149_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12149 row12149_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12149_large_left_left : List Nat := row12149_small ++ row12149_block_005
theorem row12149_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12149 row12149_large_left_left :=
  Budget.RowUpperBound.branch row12149_small_covers row12149_block_005_covers

def row12149_large_left_right_right : List Nat := row12149_block_007 ++ row12149_block_008
theorem row12149_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12149 row12149_large_left_right_right :=
  Budget.RowUpperBound.branch row12149_block_007_covers row12149_block_008_covers

def row12149_large_left_right : List Nat := row12149_block_006 ++ row12149_large_left_right_right
theorem row12149_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12149 row12149_large_left_right :=
  Budget.RowUpperBound.branch row12149_block_006_covers row12149_large_left_right_right_covers

def row12149_large_left : List Nat := row12149_large_left_left ++ row12149_large_left_right
theorem row12149_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12149 row12149_large_left :=
  Budget.RowUpperBound.branch row12149_large_left_left_covers row12149_large_left_right_covers

def row12149_large_right_left : List Nat := row12149_block_009 ++ row12149_block_010
theorem row12149_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12149 row12149_large_right_left :=
  Budget.RowUpperBound.branch row12149_block_009_covers row12149_block_010_covers

def row12149_large_right_right_right : List Nat := row12149_block_012 ++ row12149_block_013
theorem row12149_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12149 row12149_large_right_right_right :=
  Budget.RowUpperBound.branch row12149_block_012_covers row12149_block_013_covers

def row12149_large_right_right : List Nat := row12149_block_011 ++ row12149_large_right_right_right
theorem row12149_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12149 row12149_large_right_right :=
  Budget.RowUpperBound.branch row12149_block_011_covers row12149_large_right_right_right_covers

def row12149_large_right : List Nat := row12149_large_right_left ++ row12149_large_right_right
theorem row12149_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12149 row12149_large_right :=
  Budget.RowUpperBound.branch row12149_large_right_left_covers row12149_large_right_right_covers

def row12149_large : List Nat := row12149_large_left ++ row12149_large_right
theorem row12149_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12149 row12149_large :=
  Budget.RowUpperBound.branch row12149_large_left_covers row12149_large_right_covers

def row12149 : List Nat := row12149_large
theorem row12149_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12149 row12149 := row12149_large_covers

end ProjectEulerSolutions.PrimePairLarge
