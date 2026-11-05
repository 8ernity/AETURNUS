import React from "react";
import { motion, type HTMLMotionProps, type Variants } from "framer-motion";

/**
 * Top-level page container variants:
 * Orchestrates exit and staggered entry of child elements when routing/tabs change.
 */
export const pageVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.03,
    },
  },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.03,
    },
  },
  exit: {
    opacity: 0,
    y: -8,
    transition: {
      duration: 0.15,
      ease: "easeIn",
    },
  },
};

/**
 * Standard card item entry variants:
 * Matches parent's 'show' / 'visible' commands to slide up from 20px below with spring physics.
 */
export const cardItemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 24,
    },
  },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 24,
    },
  },
};

// Alias for itemVariants to match common convention
export const itemVariants = cardItemVariants;

/**
 * Parent container stagger variants:
 * Sequentially triggers children animations
 */
export const staggerContainerVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.02,
    },
  },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.02,
    },
  },
};

/**
 * Table row cascade variants:
 * Subtle vertical slide & fade-in for data grid rows
 */
export const tableRowVariants: Variants = {
  hidden: { opacity: 0, y: 14 },
  show: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 24,
    },
  },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 24,
    },
  },
};

export interface AnimatedCardProps extends HTMLMotionProps<"div"> {
  enableHover?: boolean;
  className?: string;
  children?: React.ReactNode;
}

/**
 * Reusable AnimatedCard:
 * - Slide up entry animation (y: 20 -> y: 0, opacity: 0 -> 1)
 * - Vertical lift hover effect (y: -5px) with spring physics
 * - Smooth cyan glow shadow transition
 */
export const AnimatedCard: React.FC<AnimatedCardProps> = ({
  children,
  className = "",
  enableHover = true,
  variants = cardItemVariants,
  whileHover,
  ...props
}) => {
  return (
    <motion.div
      variants={variants}
      whileHover={
        enableHover
          ? whileHover ?? {
              y: -5,
              transition: { type: "spring", stiffness: 350, damping: 25 },
            }
          : undefined
      }
      className={`transition-shadow duration-300 hover:shadow-cyan-500/20 ${className}`}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export interface StaggerContainerProps extends HTMLMotionProps<"div"> {
  className?: string;
  children?: React.ReactNode;
  staggerDelay?: number;
}

/**
 * Parent StaggerContainer wrapper:
 * Coordinates sequential appearance of nested AnimatedCard or motion items.
 */
export const StaggerContainer: React.FC<StaggerContainerProps> = ({
  children,
  className = "",
  staggerDelay = 0.05,
  variants,
  ...props
}) => {
  const containerVariants: Variants = variants ?? {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: staggerDelay,
        delayChildren: 0.02,
      },
    },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: staggerDelay,
        delayChildren: 0.02,
      },
    },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className={className}
      {...props}
    >
      {children}
    </motion.div>
  );
};
