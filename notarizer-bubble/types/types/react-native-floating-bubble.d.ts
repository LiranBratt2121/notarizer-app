declare module "react-native-floating-bubble" {
  export const FloatingBubble: {
    requestPermission(): Promise<boolean>;
    startBubble(): Promise<void>;
    hideBubble(): Promise<void>;
    onPress(callback: () => void): void;
    onRemove(callback: () => void): void;
  };
}
