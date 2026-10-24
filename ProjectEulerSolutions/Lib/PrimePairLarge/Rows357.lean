import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11939_small : List Nat := []
theorem row11939_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11939 row11939_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11939_block_005 : List Nat := []
theorem row11939_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11939 row11939_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11939_block_006 : List Nat := [11981, 12113]
def plan11939_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .possible, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 29), .first, .possible, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11939_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11939 row11939_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11939_block_006
  · decide
  · rfl

def row11939_block_007 : List Nat := [12953, 13121, 13229, 13367, 13397]
def plan11939_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .possible, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .possible, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .possible, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, .possible, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .second, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11939_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11939 row11939_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11939_block_007
  · decide
  · rfl

def row11939_block_008 : List Nat := []
theorem row11939_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11939 row11939_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11939_block_009 : List Nat := []
theorem row11939_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11939 row11939_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11939_block_010 : List Nat := []
theorem row11939_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11939 row11939_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11939_block_011 : List Nat := []
theorem row11939_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11939 row11939_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11939_block_012 : List Nat := []
theorem row11939_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11939 row11939_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11939_block_013 : List Nat := []
theorem row11939_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11939 row11939_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11939_large_left_left : List Nat := row11939_small ++ row11939_block_005
theorem row11939_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11939 row11939_large_left_left :=
  Budget.RowUpperBound.branch row11939_small_covers row11939_block_005_covers

def row11939_large_left_right_right : List Nat := row11939_block_007 ++ row11939_block_008
theorem row11939_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11939 row11939_large_left_right_right :=
  Budget.RowUpperBound.branch row11939_block_007_covers row11939_block_008_covers

def row11939_large_left_right : List Nat := row11939_block_006 ++ row11939_large_left_right_right
theorem row11939_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11939 row11939_large_left_right :=
  Budget.RowUpperBound.branch row11939_block_006_covers row11939_large_left_right_right_covers

def row11939_large_left : List Nat := row11939_large_left_left ++ row11939_large_left_right
theorem row11939_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11939 row11939_large_left :=
  Budget.RowUpperBound.branch row11939_large_left_left_covers row11939_large_left_right_covers

def row11939_large_right_left : List Nat := row11939_block_009 ++ row11939_block_010
theorem row11939_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11939 row11939_large_right_left :=
  Budget.RowUpperBound.branch row11939_block_009_covers row11939_block_010_covers

def row11939_large_right_right_right : List Nat := row11939_block_012 ++ row11939_block_013
theorem row11939_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11939 row11939_large_right_right_right :=
  Budget.RowUpperBound.branch row11939_block_012_covers row11939_block_013_covers

def row11939_large_right_right : List Nat := row11939_block_011 ++ row11939_large_right_right_right
theorem row11939_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11939 row11939_large_right_right :=
  Budget.RowUpperBound.branch row11939_block_011_covers row11939_large_right_right_right_covers

def row11939_large_right : List Nat := row11939_large_right_left ++ row11939_large_right_right
theorem row11939_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11939 row11939_large_right :=
  Budget.RowUpperBound.branch row11939_large_right_left_covers row11939_large_right_right_covers

def row11939_large : List Nat := row11939_large_left ++ row11939_large_right
theorem row11939_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11939 row11939_large :=
  Budget.RowUpperBound.branch row11939_large_left_covers row11939_large_right_covers

def row11939 : List Nat := row11939_large
theorem row11939_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11939 row11939 := row11939_large_covers

def row11941_small : List Nat := []
theorem row11941_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11941 row11941_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11941_block_005 : List Nat := []
theorem row11941_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11941 row11941_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11941_block_006 : List Nat := []
def plan11941_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .second, .second, .first, .first, (.firstFactor 31), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11941_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11941 row11941_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11941_block_006
  · decide
  · rfl

def row11941_block_007 : List Nat := [12889]
def plan11941_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .second, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 17), (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 19), .first, .second, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11941_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11941 row11941_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11941_block_007
  · decide
  · rfl

def row11941_block_008 : List Nat := []
theorem row11941_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11941 row11941_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11941_block_009 : List Nat := []
theorem row11941_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11941 row11941_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11941_block_010 : List Nat := []
theorem row11941_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11941 row11941_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11941_block_011 : List Nat := []
theorem row11941_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11941 row11941_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11941_block_012 : List Nat := []
theorem row11941_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11941 row11941_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11941_block_013 : List Nat := []
theorem row11941_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11941 row11941_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11941_large_left_left : List Nat := row11941_small ++ row11941_block_005
theorem row11941_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11941 row11941_large_left_left :=
  Budget.RowUpperBound.branch row11941_small_covers row11941_block_005_covers

def row11941_large_left_right_right : List Nat := row11941_block_007 ++ row11941_block_008
theorem row11941_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11941 row11941_large_left_right_right :=
  Budget.RowUpperBound.branch row11941_block_007_covers row11941_block_008_covers

def row11941_large_left_right : List Nat := row11941_block_006 ++ row11941_large_left_right_right
theorem row11941_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11941 row11941_large_left_right :=
  Budget.RowUpperBound.branch row11941_block_006_covers row11941_large_left_right_right_covers

def row11941_large_left : List Nat := row11941_large_left_left ++ row11941_large_left_right
theorem row11941_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11941 row11941_large_left :=
  Budget.RowUpperBound.branch row11941_large_left_left_covers row11941_large_left_right_covers

def row11941_large_right_left : List Nat := row11941_block_009 ++ row11941_block_010
theorem row11941_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11941 row11941_large_right_left :=
  Budget.RowUpperBound.branch row11941_block_009_covers row11941_block_010_covers

def row11941_large_right_right_right : List Nat := row11941_block_012 ++ row11941_block_013
theorem row11941_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11941 row11941_large_right_right_right :=
  Budget.RowUpperBound.branch row11941_block_012_covers row11941_block_013_covers

def row11941_large_right_right : List Nat := row11941_block_011 ++ row11941_large_right_right_right
theorem row11941_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11941 row11941_large_right_right :=
  Budget.RowUpperBound.branch row11941_block_011_covers row11941_large_right_right_right_covers

def row11941_large_right : List Nat := row11941_large_right_left ++ row11941_large_right_right
theorem row11941_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11941 row11941_large_right :=
  Budget.RowUpperBound.branch row11941_large_right_left_covers row11941_large_right_right_covers

def row11941_large : List Nat := row11941_large_left ++ row11941_large_right
theorem row11941_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11941 row11941_large :=
  Budget.RowUpperBound.branch row11941_large_left_covers row11941_large_right_covers

def row11941 : List Nat := row11941_large
theorem row11941_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11941 row11941 := row11941_large_covers

def row11953_small : List Nat := []
theorem row11953_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11953 row11953_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11953_block_005 : List Nat := []
theorem row11953_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11953 row11953_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11953_block_006 : List Nat := [12163]
def plan11953_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11953_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11953 row11953_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11953_block_006
  · decide
  · rfl

def row11953_block_007 : List Nat := [12487, 13003, 13723, 13789]
def plan11953_block_007 : List Budget.Choice := [(.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, (.firstFactor 11), (.firstFactor 19), .first, .first, .second, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 31), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 13), .second, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 37), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.secondFactor 19), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .possible, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11953_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11953 row11953_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11953_block_007
  · decide
  · rfl

def row11953_block_008 : List Nat := []
theorem row11953_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11953 row11953_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11953_block_009 : List Nat := []
theorem row11953_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11953 row11953_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11953_block_010 : List Nat := []
theorem row11953_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11953 row11953_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11953_block_011 : List Nat := []
theorem row11953_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11953 row11953_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11953_block_012 : List Nat := []
theorem row11953_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11953 row11953_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11953_block_013 : List Nat := []
theorem row11953_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11953 row11953_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11953_large_left_left : List Nat := row11953_small ++ row11953_block_005
theorem row11953_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11953 row11953_large_left_left :=
  Budget.RowUpperBound.branch row11953_small_covers row11953_block_005_covers

def row11953_large_left_right_right : List Nat := row11953_block_007 ++ row11953_block_008
theorem row11953_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11953 row11953_large_left_right_right :=
  Budget.RowUpperBound.branch row11953_block_007_covers row11953_block_008_covers

def row11953_large_left_right : List Nat := row11953_block_006 ++ row11953_large_left_right_right
theorem row11953_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11953 row11953_large_left_right :=
  Budget.RowUpperBound.branch row11953_block_006_covers row11953_large_left_right_right_covers

def row11953_large_left : List Nat := row11953_large_left_left ++ row11953_large_left_right
theorem row11953_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11953 row11953_large_left :=
  Budget.RowUpperBound.branch row11953_large_left_left_covers row11953_large_left_right_covers

def row11953_large_right_left : List Nat := row11953_block_009 ++ row11953_block_010
theorem row11953_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11953 row11953_large_right_left :=
  Budget.RowUpperBound.branch row11953_block_009_covers row11953_block_010_covers

def row11953_large_right_right_right : List Nat := row11953_block_012 ++ row11953_block_013
theorem row11953_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11953 row11953_large_right_right_right :=
  Budget.RowUpperBound.branch row11953_block_012_covers row11953_block_013_covers

def row11953_large_right_right : List Nat := row11953_block_011 ++ row11953_large_right_right_right
theorem row11953_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11953 row11953_large_right_right :=
  Budget.RowUpperBound.branch row11953_block_011_covers row11953_large_right_right_right_covers

def row11953_large_right : List Nat := row11953_large_right_left ++ row11953_large_right_right
theorem row11953_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11953 row11953_large_right :=
  Budget.RowUpperBound.branch row11953_large_right_left_covers row11953_large_right_right_covers

def row11953_large : List Nat := row11953_large_left ++ row11953_large_right
theorem row11953_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11953 row11953_large :=
  Budget.RowUpperBound.branch row11953_large_left_covers row11953_large_right_covers

def row11953 : List Nat := row11953_large
theorem row11953_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11953 row11953 := row11953_large_covers

def row11959_small : List Nat := []
theorem row11959_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11959 row11959_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11959_block_005 : List Nat := []
theorem row11959_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11959 row11959_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11959_block_006 : List Nat := []
def plan11959_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 37), .first]
set_option maxRecDepth 2048 in
theorem row11959_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11959 row11959_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11959_block_006
  · decide
  · rfl

def row11959_block_007 : List Nat := []
def plan11959_block_007 : List Budget.Choice := [(.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 17), .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11959_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11959 row11959_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11959_block_007
  · decide
  · rfl

def row11959_block_008 : List Nat := []
theorem row11959_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11959 row11959_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11959_block_009 : List Nat := []
theorem row11959_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11959 row11959_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11959_block_010 : List Nat := []
theorem row11959_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11959 row11959_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11959_block_011 : List Nat := []
theorem row11959_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11959 row11959_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11959_block_012 : List Nat := []
theorem row11959_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11959 row11959_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11959_block_013 : List Nat := []
theorem row11959_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11959 row11959_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11959_large_left_left : List Nat := row11959_small ++ row11959_block_005
theorem row11959_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11959 row11959_large_left_left :=
  Budget.RowUpperBound.branch row11959_small_covers row11959_block_005_covers

def row11959_large_left_right_right : List Nat := row11959_block_007 ++ row11959_block_008
theorem row11959_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11959 row11959_large_left_right_right :=
  Budget.RowUpperBound.branch row11959_block_007_covers row11959_block_008_covers

def row11959_large_left_right : List Nat := row11959_block_006 ++ row11959_large_left_right_right
theorem row11959_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11959 row11959_large_left_right :=
  Budget.RowUpperBound.branch row11959_block_006_covers row11959_large_left_right_right_covers

def row11959_large_left : List Nat := row11959_large_left_left ++ row11959_large_left_right
theorem row11959_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11959 row11959_large_left :=
  Budget.RowUpperBound.branch row11959_large_left_left_covers row11959_large_left_right_covers

def row11959_large_right_left : List Nat := row11959_block_009 ++ row11959_block_010
theorem row11959_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11959 row11959_large_right_left :=
  Budget.RowUpperBound.branch row11959_block_009_covers row11959_block_010_covers

def row11959_large_right_right_right : List Nat := row11959_block_012 ++ row11959_block_013
theorem row11959_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11959 row11959_large_right_right_right :=
  Budget.RowUpperBound.branch row11959_block_012_covers row11959_block_013_covers

def row11959_large_right_right : List Nat := row11959_block_011 ++ row11959_large_right_right_right
theorem row11959_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11959 row11959_large_right_right :=
  Budget.RowUpperBound.branch row11959_block_011_covers row11959_large_right_right_right_covers

def row11959_large_right : List Nat := row11959_large_right_left ++ row11959_large_right_right
theorem row11959_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11959 row11959_large_right :=
  Budget.RowUpperBound.branch row11959_large_right_left_covers row11959_large_right_right_covers

def row11959_large : List Nat := row11959_large_left ++ row11959_large_right
theorem row11959_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11959 row11959_large :=
  Budget.RowUpperBound.branch row11959_large_left_covers row11959_large_right_covers

def row11959 : List Nat := row11959_large
theorem row11959_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11959 row11959 := row11959_large_covers

end ProjectEulerSolutions.PrimePairLarge
